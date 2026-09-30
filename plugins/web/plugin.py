"""Fetch a web page and hand the model readable text.

Three tools, in increasing order of what they cost:

- ``web_fetch`` -- fetch a URL and return it as markdown.
- ``read_result`` -- fetch a URL and return a size summary plus the opening of
  the page, so a model can decide whether the full fetch is worth it.
- ``web_query`` -- search a public index, if an API key happens to be set.

Everything is stdlib. The harness has no third-party HTTP or HTML dependency
and a plugin that introduces one would be unusable in a checkout where
``pip install`` never ran, so this file uses ``urllib.request`` and
``html.parser`` and nothing else.

**The security work is the point.** This is a component whose whole job is to
turn a string the model read into a socket it opens. That makes every
plausible attack is "give the model a URL that is not what it looks like":

- ``file:///etc/passwd``, ``gopher://``, ``ftp://`` -- the scheme check is
  first and unconditional, before anything is resolved or opened.
- ``http://127.0.0.1:8080/admin``, ``http://169.254.169.254/latest/meta-data/``
  -- the SSRF guard resolves the name and checks *every* address it resolves
  to, because a public hostname that resolves to a private address is the
  whole trick. The check is repeated on each redirect hop, because that is
  where a guarded request becomes an unguarded one.
- A server that claims to send 2 GB -- the body is capped at 5 MB *while it is
  being read*, so the cap holds against a lying ``Content-Length`` and
  against a body that is not chunked either.
- A model looping on one host -- per-host rate limiting.

None of this makes the plugin a sandbox. A plugin with ``subprocess`` can do
anything a Python program can, and this one has ``net``, so the guards are
defence in depth and the honest statement of the limit is at the bottom of
this docstring.

**Names.** The host refuses a plugin tool that shadows a built-in, and
``web_search`` is a built-in, so the search tool here is ``web_query``. The
function is still called ``web_query`` rather than reusing the built-in's
``web_search`` so a reader can tell from the call site which implementation
they are looking at.

**Testability.** ``_FETCHER`` is the single seam. Tests replace it with a fake
and never open a socket; ``check_url`` takes the resolver and the clock as
arguments for the same reason.
"""

from __future__ import annotations

import gzip
import io
import ipaddress
import json
import os
import re
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

# --- limits ---------------------------------------------------------------

#: Hard ceiling on the compressed body, enforced while reading. A server that
#: claims 2 GB in Content-Length and then streams it is stopped at 5 MB.
MAX_RESPONSE_BYTES = 5 * 1024 * 1024

#: The decompressed body is bounded too, so a 40 KB gzip bomb cannot expand
#: into memory after passing the cap above.
MAX_DECODED_BYTES = MAX_RESPONSE_BYTES

DEFAULT_TIMEOUT = 15.0
MAX_TIMEOUT = 60.0

#: Redirect hops. Five is enough for a normal http/https/canonical chain and
#: low enough that a redirect loop is a message rather than a hang.
MAX_REDIRECTS = 5

#: Per host, within RATE_WINDOW_S. A model that has found one interesting
#: site and is trying forty URLs on it gets told no.
RATE_LIMIT_COUNT = 5
RATE_LIMIT_WINDOW_S = 60.0
#: Bound on the bookkeeping itself, so a sweep across many hosts cannot make
#: this module the memory leak.
RATE_LIMIT_MAX_HOSTS = 1024

USER_AGENT = "adaptive-harness-web-plugin/1.0 (+https://github.com/adaptive-harness)"

#: The only two schemes this plugin will open. Everything else -- file, ftp,
#: gopher, data, javascript, and any scheme invented later -- is refused
#: before a name is resolved.
ALLOWED_SCHEMES = frozenset({"http", "https"})

#: Ports that exist to reach something that is not the public internet.
_BLOCKED_PORTS = frozenset({22, 23, 25, 445, 3306, 5432, 6379, 11211, 27017})

#: Networks that are not public. Loopback, RFC1918, link-local (which includes
#: the cloud metadata endpoint at 169.254.169.254), CGNAT, benchmarking,
#: multicast, and their IPv6 equivalents.
_BLOCKED_NETWORKS: Tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = tuple(
    ipaddress.ip_network(cidr) for cidr in (
        "0.0.0.0/8",          # "this network" -- also 0.0.0.0 itself
        "10.0.0.0/8",         # RFC1918
        "100.64.0.0/10",      # CGNAT
        "127.0.0.0/8",        # loopback
        "169.254.0.0/16",     # link-local, incl. cloud metadata
        "172.16.0.0/12",      # RFC1918
        "192.0.0.0/24",       # IETF protocol assignments
        "192.168.0.0/16",     # RFC1918
        "198.18.0.0/15",      # benchmarking
        "198.51.100.0/24",    # TEST-NET-2
        "203.0.113.0/24",     # TEST-NET-3
        "224.0.0.0/4",        # multicast
        "240.0.0.0/4",        # reserved, incl. 255.255.255.255
        "::1/128",            # loopback
        "::/128",             # unspecified
        "fc00::/7",           # unique local
        "fe80::/10",          # link-local
        "ff00::/8",           # multicast
    ))


#: The address a pre-flight check pretends a search result's hostname
#: resolved to. A result row is checked for its scheme and its literal address
#: here; the real lookup happens in web_fetch, immediately before the connect.
_PUBLIC_STAND_IN = "93.184.216.34"


class WebGuardError(Exception):
    """A request refused before, or during, the network call.

    Carries the user-facing message, so the handlers can report the reason
    rather than a traceback: a refusal is information the model needs.
    """


# --- the SSRF guard -------------------------------------------------------


def _normalise_ip(address: str) -> Optional[ipaddress._BaseAddress]:
    """Parse an address, unwrapping IPv4-mapped IPv6.

    ``::ffff:127.0.0.1`` is how a dual-stack resolver can hand back loopback
    wearing an IPv6 costume, and a naive ``in ip_network("127.0.0.0/8")``
    test would pass it. ``ipv4_mapped`` unwraps it first.
    """
    try:
        parsed = ipaddress.ip_address(address.split("%", 1)[0])  # drop a zone id
    except ValueError:
        return None
    mapped = getattr(parsed, "ipv4_mapped", None)
    return mapped if mapped is not None else parsed


def is_blocked_address(address: Any) -> bool:
    """Whether a literal or resolved address must not be connected to."""
    if isinstance(address, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        parsed = address
        mapped = getattr(parsed, "ipv4_mapped", None)
        if mapped is not None:
            parsed = mapped
    else:
        parsed = _normalise_ip(str(address))
    if parsed is None:
        return True  # unparseable is not an address we are willing to dial
    if parsed.is_private or parsed.is_loopback or parsed.is_link_local \
            or parsed.is_multicast or parsed.is_reserved or parsed.is_unspecified:
        return True
    return any(parsed in network for network in _BLOCKED_NETWORKS)


def _default_resolver(host: str, port: int) -> List[str]:
    """Every address *host* resolves to. An empty list means it does not."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise WebGuardError(f"Could not resolve {host!r}: {exc.strerror or exc}.") from exc
    except (OSError, UnicodeError) as exc:  # a malformed IDN, for instance
        raise WebGuardError(f"Could not resolve {host!r}: {exc}.") from exc
    return [info[4][0] for info in infos]


def check_url(url: str, *, resolver: Optional[Callable[[str, int], Sequence[str]]] = None,
              allow_private: bool = False, resolve: bool = True) -> Tuple[str, str, int]:
    """Validate a URL and return ``(scheme, host, port)``.

    Raises :class:`WebGuardError` with a message meant for the model. Called
    once before the first request and again on every redirect hop, because a
    redirect is a fresh URL that has not been checked yet.

    ``resolve=False`` checks the scheme, the port and any literal address, and
    skips the DNS lookup. The tools pass it so that a refusal of an obviously
    bad URL never depends on reaching a resolver at all; the lookup itself
    happens once, inside the fetcher, immediately before the connect.

    ``allow_private`` exists so a user can point this at their own service on
    purpose. It is never set from tool arguments -- the model cannot grant
    itself the exception.
    """
    if not isinstance(url, str) or not url.strip():
        raise WebGuardError("No URL given.")
    candidate = url.strip()
    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        raise WebGuardError(f"{url!r} is not a URL this tool can parse: {exc}.") from exc

    scheme = (parts.scheme or "").lower()
    if not scheme:
        raise WebGuardError(
            f"{url!r} has no scheme. Give a full http:// or https:// URL -- "
            "a bare hostname is not something this tool can fetch.")
    if scheme not in ALLOWED_SCHEMES:
        raise WebGuardError(
            f"Refusing the scheme {scheme!r}. This tool only opens "
            f"{' and '.join(sorted(ALLOWED_SCHEMES))} URLs, so file://, ftp://, "
            "gopher://, data:// and the rest cannot be used to reach something "
            "that is not a web server.")

    try:
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise WebGuardError(f"{url!r} has an invalid port: {exc}.") from exc
    if not host:
        raise WebGuardError(f"{url!r} has no host.")
    host = host.strip().rstrip(".").lower()

    if not allow_private:
        if port in _BLOCKED_PORTS:
            raise WebGuardError(
                f"Refusing port {port}. It is a well-known local service port, "
                "not a web server. Use a public http(s) URL.")
        literal = _normalise_ip(host)
        if literal is not None and is_blocked_address(literal):
            raise WebGuardError(
                f"Refusing {host!r}: it is a private, loopback, link-local or "
                "reserved address, not a public host.")
        # The point of the guard: a public *name* that resolves to a private
        # *address*. Every resolved address is checked, not just the first.
        if resolve:
            lookup = resolver or _default_resolver
            try:
                resolved = list(lookup(host, port or (443 if scheme == "https" else 80)))
            except WebGuardError:
                raise
            if not resolved:
                raise WebGuardError(f"{host!r} did not resolve to any address.")
            for address in resolved:
                if is_blocked_address(address):
                    raise WebGuardError(
                        f"Refusing {host!r}: it resolves to {address}, which is a "
                        "private, loopback, link-local or reserved address. A public "
                        "name pointing inside the network is exactly what this check "
                        "exists to refuse.")
    return scheme, host, port or (443 if scheme == "https" else 80)


# --- per-host rate limiting ----------------------------------------------


class _RateLimiter:
    """Sliding window of request timestamps per host."""

    def __init__(self, *, count: int = RATE_LIMIT_COUNT, window: float = RATE_LIMIT_WINDOW_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.count = count
        self.window = window
        self.clock = clock
        self._hits: Dict[str, Deque[float]] = {}

    def recent(self, key: str) -> int:
        """How many hits this key has inside the window, for the message."""
        return len(self._hits.get(key, ()))

    def check(self, key: str) -> Optional[float]:
        """Record a hit and return the wait in seconds, or None if allowed."""
        now = self.clock()
        if len(self._hits) > RATE_LIMIT_MAX_HOSTS:
            self._hits = {host: stamps for host, stamps in self._hits.items()
                          if stamps and now - stamps[-1] < self.window}
        stamps = self._hits.setdefault(key, deque())
        while stamps and now - stamps[0] >= self.window:
            stamps.popleft()
        if len(stamps) >= self.count:
            return max(0.0, self.window - (now - stamps[0]))
        stamps.append(now)
        return None


_LIMITER = _RateLimiter()


def reset_rate_limits() -> None:
    """Drop all recorded hits. For tests, and for nothing else."""
    _LIMITER._hits.clear()


# --- the transport --------------------------------------------------------


@dataclass
class FetchResult:
    """One completed HTTP exchange, redirects already followed."""

    url: str                 # what was asked for
    final_url: str           # where the chain ended up, after any redirects
    status: int
    content_type: str
    charset: str
    body: bytes
    truncated: bool
    redirects: List[str] = field(default_factory=list)
    encoding: str = ""

    @property
    def media_type(self) -> str:
        return (self.content_type or "").split(";", 1)[0].strip().lower()


class _NoRedirect(HTTPRedirectHandler):
    """Turn every 3xx into an HTTPError so hops are counted and re-checked.

    The default handler follows redirects internally, which would mean the
    SSRF guard never sees the second URL. A redirect that is not inspected is
    a redirect that is not guarded.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _normalise_url(url: str) -> str:
    """Keep the shape of a URL, minus the pieces that only confuse a server."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc, parts.path or "/",
                       parts.query, ""))


def _read_capped(response: Any, max_bytes: int = MAX_RESPONSE_BYTES) -> Tuple[bytes, bool]:
    """Read at most ``max_bytes``, stopping the read rather than truncating.

    The cap is applied as the stream arrives, so a server that lies about its
    size, or does not declare one at all, cannot make this buffer the size it
    said it was.
    """
    buffer = bytearray()
    while True:
        chunk = response.read(64 * 1024)
        if not chunk:
            return bytes(buffer), False
        buffer.extend(chunk)
        if len(buffer) >= max_bytes:
            return bytes(buffer[:max_bytes]), True


def _read_capped_gzip(data: bytes, max_bytes: int = MAX_DECODED_BYTES) -> Tuple[bytes, bool]:
    """Decompress a capped gzip body, capping the *output* as well.

    The compressed form can be small and the expanded form enormous, so a cap
    on the wire alone is not a cap on memory.
    """
    output = bytearray()
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return bytes(output), False
            output.extend(chunk)
            if len(output) >= max_bytes:
                return bytes(output[:max_bytes]), True


def _parse_content_type(value: str) -> Tuple[str, str]:
    media = (value or "").split(";", 1)[0].strip().lower()
    charset = ""
    for part in (value or "").split(";")[1:]:
        key, _, val = part.partition("=")
        if key.strip().lower() == "charset":
            charset = val.strip().strip('"').lower()
    return media, charset or "utf-8"


def _http_fetch(url: str, *, timeout: float = DEFAULT_TIMEOUT,
                max_bytes: int = MAX_RESPONSE_BYTES,
                allow_private: bool = False,
                resolver: Optional[Callable[[str, int], Sequence[str]]] = None) -> FetchResult:
    """The real network call. Guarded, capped, redirect-limited, timed out.

    Replaced wholesale in tests: ``_FETCHER`` is the only path a tool takes.
    """
    opener = build_opener(_NoRedirect)
    current = _normalise_url(url)
    redirects: List[str] = []
    seen: List[str] = []

    for _ in range(MAX_REDIRECTS + 1):
        if current in seen:
            raise WebGuardError(
                f"Redirect loop: {current} has already been visited, and this tool "
                "does not follow a cycle.")
        seen.append(current)
        _, host, _ = check_url(current, resolver=resolver, allow_private=allow_private)
        wait = _LIMITER.check(host)
        if wait is not None:
            raise WebGuardError(
                f"Rate limit: {_LIMITER.recent(host)} requests to {host} in the last "
                f"{int(RATE_LIMIT_WINDOW_S)}s. Wait {wait:.0f}s, or fetch a different "
                "source. The limit is per host, so another site is not blocked.")

        request = Request(current, headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            "Accept-Encoding": "gzip",
        })
        try:
            with opener.open(request, timeout=timeout) as response:
                status = int(getattr(response, "status", 0) or response.getcode() or 0)
                headers = response.headers
                encoding = (headers.get("Content-Encoding") or "").strip().lower()
                body, truncated = _read_capped(response, max_bytes)
        except HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                location = exc.headers.get("Location") if exc.headers else None
                if not location:
                    raise WebGuardError(
                        f"{current} returned {exc.code} with no Location header to follow.") from exc
                target = urljoin(current, location.strip())
                if len(redirects) >= MAX_REDIRECTS:
                    raise WebGuardError(
                        f"More than {MAX_REDIRECTS} redirects starting at {url}. "
                        f"Last one pointed at {target}. This is usually a redirect loop "
                        "or a link shortener; the destination URL is safer to fetch "
                        "directly.") from exc
                redirects.append(f"{exc.code} {current} -> {target}")
                current = _normalise_url(target)
                continue  # the new URL goes back through check_url at the top
            raise WebGuardError(
                f"{current} returned HTTP {exc.code} {exc.reason}.") from exc
        except URLError as exc:
            reason = exc.reason
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise WebGuardError(
                    f"{current} did not respond within {timeout:g}s.") from exc
            raise WebGuardError(f"Could not reach {current}: {reason}.") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise WebGuardError(f"{current} did not respond within {timeout:g}s.") from exc
        except (OSError, ValueError) as exc:
            raise WebGuardError(f"Could not reach {current}: {exc}.") from exc

        if encoding in ("gzip", "x-gzip"):
            try:
                body, decoded_truncated = _read_capped_gzip(body, max_bytes)
            except (OSError, EOFError, ValueError) as exc:
                raise WebGuardError(
                    f"{current} claimed to be gzip but could not be decompressed: {exc}. "
                    "It may have been truncated.") from exc
            truncated = truncated or decoded_truncated
            encoding = "gzip"

        media_type, charset = _parse_content_type(headers.get("Content-Type", ""))
        return FetchResult(url=url, final_url=current, status=status,
                           content_type=headers.get("Content-Type", "") or "",
                           charset=charset, body=body, truncated=truncated,
                           redirects=redirects, encoding=encoding)

    raise WebGuardError(f"More than {MAX_REDIRECTS} redirects starting at {url}.")


#: The single seam every tool goes through. Tests replace it with a fake.
_FETCHER: Callable[..., FetchResult] = _http_fetch


# --- HTML to markdown -----------------------------------------------------


class _Markdown(HTMLParser):
    """A deliberately small HTML-to-markdown converter.

    "Small" is the design goal, not an apology: this is not a rendering
    engine, it is enough structure that a model reading the output can find the
    heading it wants and follow the link to the source. Anything decorative --
    scripts, styles, nav noise, inline markup -- is dropped rather than
    half-converted, because a page full of half-converted tags is worse for
    the model than a page that is simply shorter.
    """

    _SKIP = {"script", "style", "noscript", "template", "svg", "math", "head",
             "iframe", "object", "canvas", "form", "button", "select", "textarea"}
    _BLOCK = {"p", "div", "section", "article", "main", "header", "footer", "aside",
              "blockquote", "pre", "ul", "ol", "table", "tr", "hr", "figure",
              "figcaption", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
    _HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: List[str] = []
        self._skip_depth = 0
        self._pre_depth = 0
        self._link: Optional[str] = None
        self._link_text: List[str] = []

    # -- helpers

    def _emit(self, text: str) -> None:
        if self._pre_depth:
            self.parts.append(text)
        elif self._link is not None:
            self._link_text.append(text)
        else:
            self.parts.append(text)

    def _newline(self, count: int = 2) -> None:
        if self._pre_depth:
            return
        self.parts.append("\n" * count)

    # -- parser callbacks

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attributes = {key.lower(): (value or "") for key, value in attrs}
        if self._skip_depth:
            if tag in self._SKIP:
                self._skip_depth += 1
            return
        if tag in self._SKIP:
            self._skip_depth = 1
            return
        if tag in self._HEADINGS:
            self._newline(2)
            self._emit("#" * self._HEADINGS[tag] + " ")
        elif tag in ("ul", "ol"):
            self._newline(2)
        elif tag == "li":
            self._emit("- ")
        elif tag == "pre":
            self._newline(2)
            self._emit("```\n")
            self._pre_depth += 1
        elif tag == "code" and not self._pre_depth:
            self._emit("`")
        elif tag == "a":
            href = attributes.get("href", "").strip()
            self._link = self._absolute(href) if href else ""
            self._link_text = []
        elif tag == "blockquote":
            self._newline(2)
            self._emit("> ")
        elif tag == "hr":
            self._newline(2)
            self._emit("---")
            self._newline(2)
        elif tag in ("td", "th"):
            self._emit(" | ")
        elif tag in ("br",):
            self._newline(1)
        elif tag in self._BLOCK:
            self._newline(2)

    def handle_startendtag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() == "br":
            self._newline(1)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag in self._SKIP:
                self._skip_depth -= 1
            return
        if tag == "pre" and self._pre_depth:
            self._emit("\n```")
            self._pre_depth -= 1
            self._newline(2)
        elif tag == "code" and not self._pre_depth:
            self._emit("`")
        elif tag == "a" and self._link is not None:
            # Capture before clearing: emitting while the link is still open
            # would send the rendered link back into the buffer it came from.
            text = _tidy("".join(self._link_text))
            href = self._link
            self._link = None
            self._link_text = []
            if href and text and not href.lower().startswith(("javascript:", "#", "data:")):
                self.parts.append(f"[{text}]({href})")
            elif text:
                self.parts.append(text)
        elif tag == "li":
            self._newline(1)
        elif tag in ("td", "th"):
            pass  # the pipes are emitted by the start tag
        elif tag == "tr":
            self._newline(1)
        elif tag in self._HEADINGS or tag in self._BLOCK:
            self._newline(2)

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data:
            return
        self._emit(data)

    def _absolute(self, href: str) -> str:
        if not href or href.startswith("#"):
            return ""
        if not self.base_url:
            return href
        return urljoin(self.base_url, href)

    def markdown(self) -> str:
        return "".join(self.parts)


#: A line starting with one of these is markdown structure, not a wrapped
#: continuation of the paragraph above it.
_STRUCTURAL = re.compile(r"^(?:[-*+]\s|#{1,6}\s|\||>|\d+\.\s)")


def _tidy(text: str) -> str:
    """Collapse the whitespace a browser would have collapsed.

    A newline inside a paragraph is not a newline in HTML, it is a space, so
    leaving it in makes every wrapped source line look like a separate thought
    and the model reads a page as far more fragmented than it is. Fenced code
    is exempt: there, the newline and its indentation are the data.
    """
    text = unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: List[str] = []
    pending: Optional[str] = None   # a paragraph line, still open for joining
    in_code = False

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            lines.append(pending)
            pending = None

    for raw in text.split("\n"):
        if raw.lstrip().startswith("```"):
            flush()
            in_code = not in_code
            lines.append(raw.strip())
            continue
        if in_code:
            lines.append(raw.rstrip())  # indentation is meaningful here
            continue
        line = re.sub(r"[^\S\n]{2,}", " ", raw).strip()
        if not line:
            flush()
            lines.append("")
        elif pending is not None and not _STRUCTURAL.match(pending) \
                and not _STRUCTURAL.match(line):
            pending = f"{pending} {line}"
        else:
            flush()
            pending = line
    flush()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_markdown(html: str, *, base_url: str = "") -> str:
    """Convert an HTML document to readable markdown."""
    parser = _Markdown(base_url=base_url)
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup must not lose the text
        pass
    return _tidy(parser.markdown())


# --- tools ----------------------------------------------------------------

#: Bodies a model can read usefully. Anything else is reported as such rather
#: than handed over as a wall of base64 or JSON it will try to interpret.
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/xhtml"})
_TEXT_TYPES = frozenset({"text/plain", "text/markdown", "application/json",
                         "application/xml", "text/xml", "application/javascript",
                         "text/csv", "application/x-yaml", "text/yaml"})

#: Bodies large enough that handing over the whole thing is a mistake. The
#: model can ask for a bigger slice, or ask for the summary.
_BULK_TYPES = frozenset({"application/pdf", "application/octet-stream",
                         "application/zip", "application/gzip",
                         "application/x-tar", "application/msword",
                         "application/vnd.openxmlformats-officedocument"})


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    """Coerce a model-supplied number into range, tolerating a bad one."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(number, high))


def _clamp_timeout(value: Any) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT
    if seconds != seconds or seconds <= 0:  # NaN, zero, negative
        return DEFAULT_TIMEOUT
    return min(seconds, MAX_TIMEOUT)


def _fetch_or_error(url: str, *, timeout: float, max_bytes: int = MAX_RESPONSE_BYTES) -> FetchResult:
    # The scheme, the port and any literal address are refused *here*, before
    # the fetcher is called at all, so that a guard refusal never depends on
    # the fetcher being the one that was written. The name is resolved and
    # checked once more inside the fetcher, immediately before the connect.
    check_url(url, resolve=False)
    try:
        return _FETCHER(url, timeout=timeout, max_bytes=max_bytes)
    except WebGuardError:
        raise  # a refusal is already a message meant for the model
    except Exception as exc:  # noqa: BLE001 - a network library is allowed to surprise us
        raise WebGuardError(f"Fetching {url} failed: {type(exc).__name__}: {exc}") from exc


def _render(result: FetchResult, url: str, max_chars: int) -> str:
    """Turn a response into the text the model sees, plus its provenance."""
    media = result.media_type
    notes: List[str] = []
    if result.redirects:
        notes.append("redirected: " + "; ".join(result.redirects))
    if result.truncated:
        notes.append(f"the body exceeded the {MAX_RESPONSE_BYTES // (1024 * 1024)} MB cap "
                     f"and was cut at {len(result.body)} bytes")

    if media in _BULK_TYPES:
        return (f"{url} is a {media} file, {len(result.body)} bytes"
                f"{' (truncated)' if result.truncated else ''}. This tool does not convert "
                "that format. Fetch a text or HTML version, or a link to the data, and "
                "download the file with a tool that handles it.")
    if media and media not in _HTML_TYPES and media not in _TEXT_TYPES:
        # Saying so beats emitting the bytes and letting the model guess.
        return (f"{url} served {result.content_type or 'an undeclared content type'}, "
                f"{len(result.body)} bytes. This tool only turns HTML and plain text "
                "into readable markdown. Use a URL that serves text/html or text/plain, "
                "or handle the file directly.")

    body = result.body
    if media in _HTML_TYPES or not media:
        text = html_to_markdown(body.decode(result.charset or "utf-8", errors="replace"),
                                base_url=result.final_url)
        kind = "markdown"
    else:
        text = _tidy(body.decode(result.charset or "utf-8", errors="replace"))
        kind = media

    header = f"Source: {result.final_url} (HTTP {result.status}, {kind}, {len(body)} bytes"
    header += f", {'; '.join(notes)})" if notes else ")"
    if not text.strip():
        return (f"{header}\n\nThe page parsed to no text. It may be "
                "JavaScript-rendered, an image, or an empty response.")

    if len(text) > max_chars:
        return (f"{header}\n\n{text[:max_chars]}\n\n[truncated at {max_chars} characters of "
                f"{len(text)}. Ask again with a larger max_chars for more, or use "
                "read_result first to see whether the rest is worth it.]")
    return f"{header}\n\n{text}"


def web_fetch(url: str, max_chars: int = 20000) -> Dict[str, Any]:
    """Fetch a public web page and return it as markdown."""
    max_chars = _clamp(max_chars, 200, 400000, 20000)
    timeout = _clamp_timeout(os.environ.get("ADAPTIVE_HARNESS_WEB_TIMEOUT"))
    try:
        result = _fetch_or_error(url, timeout=timeout)
        return {"success": True, "output": _render(result, url, max_chars),
                "metadata": {"final_url": result.final_url, "status": result.status,
                             "content_type": result.content_type, "bytes": len(result.body),
                             "truncated": result.truncated}}
    except WebGuardError as exc:
        return {"success": False, "error": str(exc)}


def read_result(url: str, max_chars: int = 4000) -> Dict[str, Any]:
    """Summarise a page's shape before paying for the whole of it."""
    max_chars = _clamp(max_chars, 200, 100000, 4000)
    timeout = _clamp_timeout(os.environ.get("ADAPTIVE_HARNESS_WEB_TIMEOUT"))
    try:
        result = _fetch_or_error(url, timeout=timeout)
    except WebGuardError as exc:
        return {"success": False, "error": str(exc)}

    media = result.media_type
    convertible = media in _HTML_TYPES or media in _TEXT_TYPES or not media
    text = ""
    if convertible:
        if media in _HTML_TYPES or not media:
            text = html_to_markdown(result.body.decode(result.charset or "utf-8",
                                                       errors="replace"),
                                    base_url=result.final_url)
        else:
            text = _tidy(result.body.decode(result.charset or "utf-8", errors="replace"))

    lines = [
        f"URL: {result.final_url}" + (f" (redirected from {url})" if result.final_url != url else ""),
        f"Status: HTTP {result.status}",
        f"Type: {result.content_type or 'undeclared'}",
        f"Size: {len(result.body)} bytes"
        + (" (cut at the size cap)" if result.truncated else ""),
        f"Readable text: {len(text)} characters"
        + ("" if convertible else " (not convertible by this tool)"),
    ]
    if result.redirects:
        lines.append("Redirect chain: " + " -> ".join(result.redirects))
    preview = text[:max_chars].strip()
    if preview:
        lines += ["", "--- opening of the page ---", preview]
        if len(text) > max_chars:
            lines.append(f"[... {len(text) - max_chars} more characters. "
                         f"Call web_fetch with this URL for up to 20000 at a time.]")
    return {"success": True, "output": "\n".join(lines),
            "metadata": {"final_url": result.final_url, "status": result.status,
                         "content_type": result.content_type, "bytes": len(result.body),
                         "text_chars": len(text)}}


def _search_api_key() -> str:
    """The key, read at call time.

    Read inside the function rather than at import, so installing the plugin
    never requires configuring a service. ``BRAVE_SEARCH_API_KEY`` matches the
    built-in ``web_search`` tool, so one key serves both.
    """
    return (os.environ.get("BRAVE_SEARCH_API_KEY")
            or os.environ.get("ADAPTIVE_HARNESS_SEARCH_API_KEY") or "").strip()


def web_query(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search a public index. Requires an API key in the environment.

    Named ``web_query`` rather than ``web_search`` because the harness already
    ships a built-in tool by that name and the plugin host refuses to let a
    plugin shadow a built-in.
    """
    if not isinstance(query, str) or not query.strip():
        return {"success": False, "error": "The search query cannot be empty."}
    limit = _clamp(limit, 1, 20, 5)

    key = _search_api_key()
    if not key:
        return {"success": False, "error": (
            "No search API key is configured, so this tool cannot search. Set "
            "BRAVE_SEARCH_API_KEY (or ADAPTIVE_HARNESS_SEARCH_API_KEY) in the "
            "environment to enable it. This is a configuration gap, not a failure "
            "to try: nothing was sent anywhere. If you already know a URL, web_fetch "
            "works without a key.")}

    endpoint = os.environ.get("ADAPTIVE_HARNESS_SEARCH_ENDPOINT",
                              "https://api.search.brave.com/res/v1/web/search")
    url = f"{endpoint}?{urlencode({'q': query.strip(), 'count': limit})}"
    timeout = _clamp_timeout(os.environ.get("ADAPTIVE_HARNESS_WEB_TIMEOUT"))

    request_headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if "brave" in endpoint.lower():
        request_headers["X-Subscription-Token"] = key
    else:
        # A generic endpoint: Bearer is the near-universal convention.
        request_headers["Authorization"] = f"Bearer {key}"

    try:
        # The search endpoint is third-party, so it goes through the same
        # guard as any other host. A redirect from the search API must not be
        # a way to reach something private.
        _, host, _ = check_url(url)
        wait = _LIMITER.check(host)
        if wait is not None:
            return {"success": False, "error": f"Rate limit: wait {wait:.0f}s before searching again."}
        from urllib.request import urlopen  # local: keeps the module's net surface obvious
        request = Request(url, headers=request_headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                body, truncated = _read_capped(response, 1024 * 1024)
        except HTTPError as exc:
            return {"success": False, "error": (
                f"The search API returned HTTP {exc.code} {exc.reason}."
                + (" The key is probably wrong or out of quota." if exc.code in (401, 403)
                   else ""))}
        except URLError as exc:
            return {"success": False, "error": f"Could not reach the search API: {exc.reason}."}
        except (TimeoutError, socket.timeout):
            return {"success": False, "error": f"The search API did not respond within {timeout:g}s."}
        except (OSError, ValueError) as exc:
            return {"success": False, "error": f"Could not reach the search API: {exc}."}

        try:
            payload = json.loads(body.decode("utf-8", errors="replace"))
        except ValueError as exc:
            return {"success": False, "error": f"The search API did not return JSON: {exc}."}
    except WebGuardError as exc:
        return {"success": False, "error": str(exc)}

    results = _parse_search_results(payload, limit)
    if not results:
        return {"success": True, "output": f"No results for {query.strip()!r}. Try broader terms.",
                "metadata": {"result_count": 0, "source_urls": []}}

    lines: List[str] = []
    urls: List[str] = []
    for index, item in enumerate(results, start=1):
        lines.append(f"{index}. {item['title']}\n   {item['url']}\n   {item['snippet']}".rstrip())
        urls.append(item["url"])
    note = " (response was cut at the size cap)" if truncated else ""
    return {"success": True, "output": f"Results for {query.strip()!r}{note}:\n\n" + "\n\n".join(lines),
            "metadata": {"result_count": len(results), "source_urls": urls}}


def _parse_search_results(payload: Any, limit: int) -> List[Dict[str, str]]:
    """Pull results out of the common search-API shapes, and check each URL.

    A result URL is attacker-influenced text, so it is put through the guard.
    That is not paranoia: a search index that returned an internal address
    would be a way to route the model into ``web_fetch`` against the very
    network it just refused. Resolution is stubbed here to a public address,
    because the check that matters for a result row is the one on the literal
    address and the scheme -- and ``web_fetch`` resolves and re-checks the name
    for real before it connects.
    """
    items: Iterable[Any] = ()
    if isinstance(payload, dict):
        web = payload.get("web")
        if isinstance(web, dict) and isinstance(web.get("results"), list):
            items = web["results"]
        elif isinstance(payload.get("results"), list):
            items = payload["results"]
        elif isinstance(payload.get("items"), list):
            items = payload["items"]
    results: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        target = str(item.get("url") or item.get("link") or "").strip()
        if not target:
            continue
        try:
            check_url(target, resolver=lambda host, port: [_PUBLIC_STAND_IN])
        except WebGuardError:
            # The row is dropped, and the address is not echoed back: a URL
            # printed into the context is a URL something else may act on.
            # One bad row must not abort the whole search, either.
            results.append({"title": "[result withheld]", "url": "(not shown)",
                            "snippet": "withheld: this result points at a private "
                                       "address or a scheme that is not http(s)"})
            continue
        results.append({
            "title": _tidy(str(item.get("title") or item.get("name") or "Untitled")),
            "url": target,
            "snippet": _tidy(str(item.get("description") or item.get("snippet") or "")),
        })
        if len(results) >= limit:
            break
    return results
