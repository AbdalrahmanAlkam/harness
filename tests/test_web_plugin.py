"""The bundled web plugin: the guard, the cap, and the conversion.

Nothing here touches the network. Every test that would otherwise open a
socket gets a fake: ``_FETCHER`` is replaced with a stub, ``build_opener``
and ``urllib.request.urlopen`` are replaced with callables that record being
called, and ``check_url`` is handed its resolver and its clock. The point of
these tests is the refusal paths, so a suite that could succeed by reaching
the internet would be testing the wrong thing.
"""

from __future__ import annotations

import email.message
import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

BUNDLED_DIR = (Path(__file__).resolve().parents[1] / "src" / "adaptive_harness"
               / "plugins" / "bundled" / "web")
MANIFEST_PATH = BUNDLED_DIR / "plugin.plugin.json"

#: A public address, so a test that is about something other than resolution
#: cannot accidentally pass because the fake resolver returned loopback.
PUBLIC_IP = "93.184.216.34"
PUBLIC_RESOLVER = lambda host, port: [PUBLIC_IP]  # noqa: E731 - a stub, not a lambda factory


def _load_module():
    """Import the plugin file the way the host does, under a private name.

    The module goes into ``sys.modules`` first, as the host does it: without
    that, ``dataclasses`` cannot resolve the postponed annotations in the
    file and the import fails for a reason that has nothing to do with the
    code under test.
    """
    spec = importlib.util.spec_from_file_location(
        "web_plugin_under_test", BUNDLED_DIR / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def web():
    module = _load_module()
    module.reset_rate_limits()
    yield module
    module.reset_rate_limits()


def result(web, *, url="https://example.com/page", status=200,
           content_type="text/html; charset=utf-8", body=b"<p>hi</p>",
           truncated=False, final_url=None, redirects=()):
    """A FetchResult, built without going near a socket."""
    return web.FetchResult(
        url=url, final_url=final_url or url, status=status,
        content_type=content_type, charset=content_type.split("charset=")[-1]
        if "charset=" in content_type else "utf-8",
        body=body, truncated=truncated, redirects=list(redirects))


# --- the SSRF guard: literal private and loopback addresses ---------------

PRIVATE_TARGETS = [
    "http://127.0.0.1/",
    "http://127.0.0.1:8080/admin",
    "http://127.5.5.5/",
    "http://0.0.0.0/",
    "http://10.0.0.1/",
    "http://10.255.255.254/",
    "http://172.16.0.1/",
    "http://172.31.255.254/",
    "http://192.168.1.1/",
    "http://192.168.0.1:5000/",
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://[::1]/",
    "http://[::1]:9000/",
    "http://[fc00::1]/",
    "http://[fd12:3456::1]/",
    "http://[fe80::1]/",
    "http://[::ffff:127.0.0.1]/",       # loopback wearing an IPv6 costume
    "http://[::ffff:169.254.169.254]/",
    "http://255.255.255.255/",
    "http://100.64.0.1/",               # CGNAT
    "http://198.18.0.1/",               # benchmarking
    "http://[::]/",
    "http://192.0.2.1/",                # TEST-NET, reserved for documentation
]


@pytest.mark.parametrize("url", PRIVATE_TARGETS)
def test_private_and_reserved_addresses_are_refused(web, url):
    with pytest.raises(web.WebGuardError) as caught:
        web.check_url(url, resolver=PUBLIC_RESOLVER)
    message = str(caught.value)
    assert "Refusing" in message
    # The message is shown to the model, so it has to explain the refusal.
    assert "private" in message or "link-local" in message or "reserved" in message


@pytest.mark.parametrize("address,blocked", [
    ("127.0.0.1", True),
    ("127.1.2.3", True),
    ("10.1.2.3", True),
    ("172.16.0.1", True),
    ("172.31.255.255", True),
    ("172.32.0.1", False),      # just outside 172.16/12
    ("192.168.0.1", True),
    ("169.254.169.254", True),
    ("0.0.0.0", True),
    ("255.255.255.255", True),
    ("::1", True),
    ("fc00::1", True),
    ("fdff::1", True),
    ("fe80::1", True),
    ("::ffff:127.0.0.1", True),
    ("::ffff:10.0.0.1", True),
    ("2001:4860:4860::8888", False),
    ("93.184.216.34", False),
    ("8.8.8.8", False),
    ("not-an-address", True),   # unparseable is not dialable
])
def test_is_blocked_address(web, address, blocked):
    assert web.is_blocked_address(address) is blocked


# --- the SSRF guard: a public name that resolves to a private address -----


@pytest.mark.parametrize("url,resolved", [
    ("http://localhost:8000/", "127.0.0.1"),
    ("http://localhost/", "::1"),
    ("http://internal.corp/", "10.0.0.5"),
    ("http://metadata.google.internal/computeMetadata/v1/", "169.254.169.254"),
    ("https://evil.example/", "127.0.0.1"),
    ("https://rebind.example/", "192.168.1.1"),
    ("http://dual-stack.example/", "fc00::1"),
    ("https://also-dual.example/", "::ffff:10.0.0.1"),
])
def test_a_public_name_resolving_to_a_private_address_is_refused(web, url, resolved):
    def resolver(host, port):
        return [PUBLIC_IP, resolved]  # a public answer and a private one

    with pytest.raises(web.WebGuardError) as caught:
        web.check_url(url, resolver=resolver)
    assert resolved in str(caught.value)


def test_a_resolvable_public_url_passes(web):
    assert web.check_url("https://example.com/docs", resolver=PUBLIC_RESOLVER) == (
        "https", "example.com", 443)
    assert web.check_url("http://example.com:8080/docs", resolver=PUBLIC_RESOLVER) == (
        "http", "example.com", 8080)


def test_a_name_that_does_not_resolve_is_an_error_not_a_fetch(web):
    with pytest.raises(web.WebGuardError) as caught:
        web.check_url("https://nowhere.invalid/", resolver=lambda host, port: [])
    assert "did not resolve" in str(caught.value)


def test_local_service_ports_are_refused(web):
    with pytest.raises(web.WebGuardError) as caught:
        web.check_url("http://database.example:5432/", resolver=PUBLIC_RESOLVER)
    assert "5432" in str(caught.value)


def test_the_guard_never_dials_when_the_literal_is_already_private(web):
    """A literal private address is refused without a lookup."""
    def explode(host, port):
        raise AssertionError("resolved a name for an address literal")

    with pytest.raises(web.WebGuardError):
        web.check_url("http://127.0.0.1/", resolver=explode)


# --- schemes --------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "file://localhost/etc/shadow",
    "ftp://ftp.example.com/pub/",
    "gopher://gopher.example.com/1",
    "data:text/html;base64,PHNjcmlwdD4=",
    "javascript:alert(1)",
    "chrome://settings",
    "ws://example.com/socket",
    "//example.com/protocol-relative",
    "example.com/no-scheme",
    "httpsx://example.com/",
    "",
    "   ",
])
def test_non_http_schemes_are_refused(web, url):
    with pytest.raises(web.WebGuardError) as caught:
        web.check_url(url, resolver=PUBLIC_RESOLVER)
    message = str(caught.value)
    assert "scheme" in message or "No URL" in message


def test_the_scheme_refusal_names_the_schemes_it_refuses(web):
    with pytest.raises(web.WebGuardError) as caught:
        web.check_url("file:///etc/passwd", resolver=PUBLIC_RESOLVER)
    assert "http" in str(caught.value) and "https" in str(caught.value)


def test_scheme_case_and_default_ports_are_normalised(web):
    assert web.check_url("HTTPS://Example.COM/Path", resolver=PUBLIC_RESOLVER) == (
        "https", "example.com", 443)


def test_a_refused_url_never_reaches_the_fetcher(web, monkeypatch):
    """The tool checks the scheme and the literal address before the call."""
    calls = []
    monkeypatch.setattr(web, "_FETCHER", lambda *a, **k: calls.append(a) or None)

    for url in ("file:///etc/passwd", "http://127.0.0.1/", "http://169.254.169.254/"):
        outcome = web.web_fetch(url)
        assert outcome["success"] is False
        assert outcome["error"]

    assert calls == []


# --- the size cap ---------------------------------------------------------


class _ChunkStream:
    """A response body that records how much of itself was actually read."""

    def __init__(self, total: int) -> None:
        self.remaining = total
        self.served = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.remaining
        chunk = min(size, self.remaining)
        self.remaining -= chunk
        self.served += chunk
        return b"x" * chunk


def test_the_cap_is_five_megabytes(web):
    assert web.MAX_RESPONSE_BYTES == 5 * 1024 * 1024


def test_read_capped_returns_a_short_body_whole(web):
    body, truncated = web._read_capped(_ChunkStream(1000), max_bytes=5000)
    assert truncated is False
    assert len(body) == 1000


def test_read_capped_cuts_at_the_cap_and_stops_reading(web):
    """The cap holds against a lying Content-Length, because the read stops."""
    stream = _ChunkStream(50 * 1024 * 1024)
    body, truncated = web._read_capped(stream, max_bytes=4096)
    assert truncated is True
    assert len(body) == 4096
    # It stopped at the cap rather than draining a body it was told about.
    assert stream.served <= 4096 + 64 * 1024
    assert stream.remaining > 0


def test_read_capped_handles_an_exactly_full_body(web):
    body, truncated = web._read_capped(_ChunkStream(4096), max_bytes=4096)
    assert len(body) == 4096
    assert truncated is True  # at the cap counts as cut, which is the honest report


def test_read_capped_handles_an_empty_body(web):
    body, truncated = web._read_capped(_ChunkStream(0), max_bytes=4096)
    assert body == b""
    assert truncated is False


def test_a_gzip_bomb_is_capped_after_decompression(web):
    """40 KB compressed, 40 MB expanded: the cap has to apply to the output."""
    payload = gzip.compress(b"A" * (40 * 1024 * 1024))
    assert len(payload) < 200 * 1024, "the test payload is supposed to be small"
    body, truncated = web._read_capped_gzip(payload, max_bytes=64 * 1024)
    assert truncated is True
    assert len(body) == 64 * 1024


def test_gzip_round_trips_when_it_fits(web):
    payload = gzip.compress(b"hello" * 1000)
    body, truncated = web._read_capped_gzip(payload, max_bytes=64 * 1024)
    assert truncated is False
    assert body == b"hello" * 1000


def test_a_truncated_response_says_so_rather_than_looking_complete(web, monkeypatch):
    monkeypatch.setattr(web, "_FETCHER", lambda *a, **k: result(
        web, body=b"<p>" + b"x" * 900, truncated=True))
    outcome = web.web_fetch("https://example.com/big")
    assert outcome["success"] is True
    assert outcome["metadata"]["truncated"] is True
    assert "5 MB cap" in outcome["output"]


# --- HTML to markdown -----------------------------------------------------

SAMPLE_HTML = """
<!doctype html>
<html>
  <head>
    <title>Ignored</title>
    <style>body { color: red; } .x { display: none }</style>
    <script>var tracking = 1; if (tracking) { alert("no"); }</script>
  </head>
  <body>
    <h1>The Title</h1>
    <p>A paragraph   with
       collapsed   whitespace and an &amp; entity.</p>
    <h2>Section</h2>
    <ul>
      <li>first item</li>
      <li>second item</li>
    </ul>
    <p>See <a href="/docs/page">the docs</a> and
       <a href="https://other.example/x">an external link</a>, or
       <a href="javascript:evil()">this</a>.</p>
    <pre><code>def f():
    return 1
</code></pre>
    <blockquote>Quoted material.</blockquote>
    <table>
      <tr><th>Name</th><th>Value</th></tr>
      <tr><td>alpha</td><td>1</td></tr>
    </table>
    <script>console.log("trailing noise")</script>
    <noscript>enable javascript</noscript>
  </body>
</html>
"""


@pytest.fixture()
def markdown(web):
    return web.html_to_markdown(SAMPLE_HTML, base_url="https://example.com/index.html")


def test_headings_become_markdown_headings(markdown):
    assert "# The Title" in markdown
    assert "## Section" in markdown


def test_paragraphs_are_joined_and_whitespace_collapsed(markdown):
    assert "A paragraph with collapsed whitespace and an & entity." in markdown
    prose = re.sub(r"```.*?```", "", markdown, flags=re.S)  # code keeps its spacing
    assert "  " not in prose, "runs of spaces should be collapsed outside code"


def test_lists_become_dashes(markdown):
    assert "- first item" in markdown
    assert "- second item" in markdown


def test_relative_links_are_resolved_and_absolute_links_kept(markdown):
    assert "[the docs](https://example.com/docs/page)" in markdown
    assert "[an external link](https://other.example/x)" in markdown


def test_javascript_links_keep_their_text_and_not_their_scheme(markdown):
    assert "javascript:evil()" not in markdown
    assert "this" in markdown


def test_code_becomes_a_fenced_block(markdown):
    assert "```" in markdown
    body = markdown.split("```")[1]
    assert "def f():" in body
    assert "return 1" in body


def test_blockquotes_are_marked(markdown):
    assert "> Quoted material." in markdown


def test_table_rows_become_pipes(markdown):
    assert "Name | Value" in markdown
    assert "alpha | 1" in markdown


def test_scripts_styles_and_noscript_content_are_gone(markdown):
    assert "tracking" not in markdown
    assert "color: red" not in markdown
    assert "console.log" not in markdown
    assert "enable javascript" not in markdown


def test_no_html_tags_survive_into_the_output(markdown):
    for tag in ("<script", "<style", "<p>", "<li>", "<h1", "</", "<table"):
        assert tag not in markdown, f"{tag!r} leaked into the markdown"


def test_markdown_has_no_triple_blank_lines(markdown):
    assert "\n\n\n\n" not in markdown


def test_conversion_survives_truncated_markup(web):
    """A response cut mid-tag still yields the text that was there."""
    broken = SAMPLE_HTML[:1200]
    out = web.html_to_markdown(broken, base_url="https://example.com/")
    assert "The Title" in out
    assert "<" not in out.split("\n")[0]


def test_conversion_of_empty_input_is_empty(web):
    assert web.html_to_markdown("") == ""
    assert web.html_to_markdown("<p></p>") == ""


def test_links_without_a_base_url_are_left_relative(web):
    out = web.html_to_markdown('<a href="/x">y</a>')
    assert out == "[y](/x)"


# --- the tools ------------------------------------------------------------


def test_web_fetch_returns_markdown_with_provenance(web, monkeypatch):
    monkeypatch.setattr(web, "_FETCHER", lambda url, **k: result(
        web, url=url, final_url=url, body=b"<h1>Doc</h1><p>Body text.</p>"))
    outcome = web.web_fetch("https://example.com/doc")
    assert outcome["success"] is True
    assert "# Doc" in outcome["output"]
    assert "https://example.com/doc" in outcome["output"]
    assert "HTTP 200" in outcome["output"]
    assert outcome["metadata"]["status"] == 200


def test_web_fetch_reports_the_final_url_after_a_redirect(web, monkeypatch):
    monkeypatch.setattr(web, "_FETCHER", lambda *a, **k: result(
        web, url="https://example.com/old", final_url="https://example.com/new",
        body=b"<p>moved</p>", redirects=["301 https://example.com/old -> https://example.com/new"]))
    outcome = web.web_fetch("https://example.com/old")
    assert "https://example.com/new" in outcome["output"]
    assert "redirected" in outcome["output"]


def test_web_fetch_respects_max_chars(web, monkeypatch):
    monkeypatch.setattr(web, "_FETCHER", lambda *a, **k: result(
        web, body=b"<p>" + b"word " * 5000 + b"</p>"))
    outcome = web.web_fetch("https://example.com/long", max_chars=1000)
    assert "truncated at 1000 characters" in outcome["output"]
    # The refusal to give more is stated, not just silently obeyed.
    assert "larger max_chars" in outcome["output"]


def test_web_fetch_on_a_binary_type_says_so_rather_than_emitting_the_bytes(web, monkeypatch):
    monkeypatch.setattr(web, "_FETCHER", lambda *a, **k: result(
        web, content_type="application/pdf", body=b"%PDF-1.7\nbinary"))
    outcome = web.web_fetch("https://example.com/paper.pdf")
    assert outcome["success"] is True
    assert "application/pdf" in outcome["output"]
    assert "%PDF" not in outcome["output"]


def test_web_fetch_on_json_returns_the_text_not_a_message(web, monkeypatch):
    monkeypatch.setattr(web, "_FETCHER", lambda *a, **k: result(
        web, content_type="application/json", body=b'{"a": 1}'))
    outcome = web.web_fetch("https://example.com/data.json")
    assert '{"a": 1}' in outcome["output"]


def test_web_fetch_reports_a_guard_failure_without_a_traceback(web, monkeypatch):
    def refuse(url, **kwargs):
        raise web.WebGuardError("Refusing 'http': scheme is not http(s).")

    monkeypatch.setattr(web, "_FETCHER", refuse)
    outcome = web.web_fetch("https://example.com")
    assert outcome["success"] is False
    assert outcome["error"].startswith("Refusing")
    assert "Traceback" not in outcome["error"]


def test_web_fetch_turns_an_unexpected_network_error_into_a_message(web, monkeypatch):
    def explode(url, **kwargs):
        raise OSError("connection reset by peer")

    monkeypatch.setattr(web, "_FETCHER", explode)
    outcome = web.web_fetch("https://example.com")
    assert outcome["success"] is False
    assert "OSError" in outcome["error"]
    assert "Traceback" not in outcome["error"]


def test_read_result_summarises_size_before_content(web, monkeypatch):
    body = ("<p>" + "filler sentence. " * 400 + "</p>").encode()
    monkeypatch.setattr(web, "_FETCHER", lambda *a, **k: result(web, body=body))
    outcome = web.read_result("https://example.com/article", max_chars=500)
    assert outcome["success"] is True
    assert "HTTP 200" in outcome["output"]
    assert "text/html" in outcome["output"]
    assert f"Size: {len(body)} bytes" in outcome["output"]
    assert "Readable text:" in outcome["output"]
    assert "filler sentence." in outcome["output"]
    # The summary comes first, so a model can decide before reading on.
    assert outcome["output"].index("Status:") < outcome["output"].index("opening of the page")
    assert outcome["metadata"]["text_chars"] > 500


def test_read_result_reports_a_refusal(web, monkeypatch):
    monkeypatch.setattr(web, "_FETCHER", lambda *a, **k: result(web, body=b"<p>x</p>"))
    # No fetcher needed: a private target is refused before it is called.
    assert web.read_result("http://169.254.169.254/")["success"] is False


def test_read_result_on_a_bulk_file_says_it_is_not_convertible(web, monkeypatch):
    monkeypatch.setattr(web, "_FETCHER", lambda *a, **k: result(
        web, content_type="application/zip", body=b"PK\x03\x04"))
    outcome = web.read_result("https://example.com/a.zip")
    assert "not convertible" in outcome["output"]


# --- search ---------------------------------------------------------------


def test_search_without_a_key_explains_and_does_nothing(web, monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("ADAPTIVE_HARNESS_SEARCH_API_KEY", raising=False)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("search opened a connection with no key"))

    outcome = web.web_query("adaptive agent harness")
    assert outcome["success"] is False
    assert "BRAVE_SEARCH_API_KEY" in outcome["error"]
    assert "No search API key is configured" in outcome["error"]


def test_search_does_not_require_the_key_at_import_time(web, monkeypatch):
    """Importing and listing the plugin must work with no key anywhere."""
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("ADAPTIVE_HARNESS_SEARCH_API_KEY", raising=False)
    module = _load_module()  # must not raise
    assert callable(module.web_query)
    assert module._search_api_key() == ""


def test_search_rejects_an_empty_query_before_checking_the_key(web, monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    assert web.web_query("   ")["success"] is False
    assert "empty" in web.web_query("   ")["error"]


@pytest.fixture()
def offline_resolver(web, monkeypatch):
    """Let the search tests exercise the guard without needing a resolver."""
    monkeypatch.setattr(web, "_default_resolver", lambda host, port: [PUBLIC_IP])
    return PUBLIC_IP


def _fake_json_response(payload):
    class _Response:
        """Streams once, like a real body: the cap loop reads until it is empty."""
        headers = email.message.Message()

        def __init__(self):
            self._body = json.dumps(payload).encode()

        def read(self, size=-1):
            chunk, self._body = self._body[:size], self._body[size:]
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Response()


def test_search_with_a_key_returns_results(web, monkeypatch, offline_resolver):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    import urllib.request

    def fake_urlopen(request, timeout=None):
        assert "X-Subscription-Token" in request.headers or \
            request.get_header("X-subscription-token") == "test-key"
        return _fake_json_response({"web": {"results": [
            {"title": "First", "url": "https://example.com/1", "description": "One."},
            {"title": "Second", "url": "https://example.com/2", "description": "Two."},
        ]}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    outcome = web.web_query("harness", limit=2)
    assert outcome["success"] is True
    assert "First" in outcome["output"]
    assert "https://example.com/1" in outcome["output"]
    assert outcome["metadata"]["result_count"] == 2


def test_search_withholds_a_result_that_points_inside_the_network(web, monkeypatch, offline_resolver):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None:
                        _fake_json_response({"web": {"results": [
                            {"title": "Local admin", "url": "http://169.254.169.254/latest/",
                             "description": "metadata"},
                            {"title": "Fine", "url": "https://example.com/ok",
                             "description": "public"},
                        ]}}))
    outcome = web.web_query("metadata")
    assert "169.254.169.254" not in outcome["output"]
    assert "withheld" in outcome["output"]
    assert "https://example.com/ok" in outcome["output"]


def test_search_reports_an_http_error_from_the_api(web, monkeypatch, offline_resolver):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "bad-key")
    import urllib.request

    def fake_urlopen(request, timeout=None):
        raise HTTPError(request.full_url, 401, "Unauthorized", email.message.Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    outcome = web.web_query("anything")
    assert outcome["success"] is False
    assert "401" in outcome["error"]


def test_search_reports_a_timeout_from_the_api(web, monkeypatch, offline_resolver):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    import urllib.request

    def fake_urlopen(request, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    outcome = web.web_query("anything")
    assert outcome["success"] is False
    assert "did not respond" in outcome["error"]


def test_search_reports_non_json_from_the_api(web, monkeypatch, offline_resolver):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    import urllib.request

    class _HTML:
        def __init__(self):
            self._body = b"<html>login page</html>"

        def read(self, size=-1):
            chunk, self._body = self._body[:size], self._body[size:]
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: _HTML())
    outcome = web.web_query("anything")
    assert outcome["success"] is False
    assert "JSON" in outcome["error"]


# --- redirects and the guard ---------------------------------------------


class _FakeOpener:
    """Stands in for urllib's opener, one scripted response at a time."""

    def __init__(self, script):
        self.script = list(script)
        self.requested = []

    def open(self, request, timeout=None):
        self.requested.append(request.full_url)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture()
def unlimited(web, monkeypatch):
    """A limiter that never fires, for tests that are not about rate limiting."""
    monkeypatch.setattr(web, "_LIMITER", web._RateLimiter(count=1000, window=60.0))
    return web._LIMITER


def _redirect(location, code=302):
    headers = email.message.Message()
    headers["Location"] = location
    return HTTPError("https://example.com/hop", code, "Found", headers, None)


def test_a_redirect_to_a_private_address_is_refused(web, monkeypatch):
    """The second URL of a chain is a fresh URL and gets the full check."""
    opener = _FakeOpener([_redirect("http://169.254.169.254/latest/meta-data/")])
    monkeypatch.setattr(web, "build_opener", lambda *a, **k: opener)

    with pytest.raises(web.WebGuardError) as caught:
        web._http_fetch("https://example.com/start")
    assert "169.254.169.254" in str(caught.value)
    assert len(opener.requested) == 1, "it must not connect to the redirect target"


def test_a_redirect_to_a_non_http_scheme_is_refused(web, monkeypatch):
    opener = _FakeOpener([_redirect("file:///etc/passwd")])
    monkeypatch.setattr(web, "build_opener", lambda *a, **k: opener)
    with pytest.raises(web.WebGuardError) as caught:
        web._http_fetch("https://example.com/start")
    assert "scheme" in str(caught.value)


def test_a_redirect_loop_is_refused(web, monkeypatch, unlimited):
    opener = _FakeOpener([_redirect("https://example.com/a")] * 20)
    monkeypatch.setattr(web, "build_opener", lambda *a, **k: opener)
    with pytest.raises(web.WebGuardError) as caught:
        web._http_fetch("https://example.com/a")
    assert "redirect" in str(caught.value).lower()
    assert web.MAX_REDIRECTS == 5


def test_too_many_redirects_stops_at_the_limit(web, monkeypatch, unlimited):
    opener = _FakeOpener([_redirect(f"https://example.com/{n}") for n in range(20)])
    monkeypatch.setattr(web, "build_opener", lambda *a, **k: opener)
    with pytest.raises(web.WebGuardError) as caught:
        web._http_fetch("https://example.com/start")
    assert "redirects" in str(caught.value)
    assert len(opener.requested) == 6, "five hops allowed, the sixth refused"


def test_a_redirect_without_a_location_is_reported(web, monkeypatch):
    headers = email.message.Message()
    opener = _FakeOpener([HTTPError("https://example.com/", 302, "Found", headers, None)])
    monkeypatch.setattr(web, "build_opener", lambda *a, **k: opener)
    with pytest.raises(web.WebGuardError) as caught:
        web._http_fetch("https://example.com/")
    assert "Location" in str(caught.value)


def test_a_real_redirect_chain_is_followed_and_reported(web, monkeypatch):
    class _OK:
        def __init__(self, body, content_type="text/html"):
            self.headers = email.message.Message()
            self.headers["Content-Type"] = content_type
            self._body = body
            self.status = 200

        def read(self, size=-1):
            chunk, self._body = self._body[:size], self._body[size:]
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    opener = _FakeOpener([_redirect("https://example.com/final"), _OK(b"<p>arrived</p>")])
    monkeypatch.setattr(web, "build_opener", lambda *a, **k: opener)
    monkeypatch.setattr(web, "check_url", lambda url, **k: (
        "https", url.split("/")[2], 443))  # keep the guard out of a DNS call

    outcome = web._http_fetch("https://example.com/start")
    assert outcome.final_url == "https://example.com/final"
    assert len(outcome.redirects) == 1
    assert b"arrived" in outcome.body


def test_an_http_error_is_a_message_not_a_traceback(web, monkeypatch):
    opener = _FakeOpener([HTTPError("https://example.com/", 503, "Service Unavailable",
                                    email.message.Message(), None)])
    monkeypatch.setattr(web, "build_opener", lambda *a, **k: opener)
    with pytest.raises(web.WebGuardError) as caught:
        web._http_fetch("https://example.com/")
    assert "503" in str(caught.value)


def test_a_timeout_is_a_message(web, monkeypatch):
    opener = _FakeOpener([TimeoutError("timed out")])
    monkeypatch.setattr(web, "build_opener", lambda *a, **k: opener)
    with pytest.raises(web.WebGuardError) as caught:
        web._http_fetch("https://example.com/")
    assert "did not respond" in str(caught.value)


# --- rate limiting and timeouts -------------------------------------------


def test_the_limiter_allows_a_burst_and_then_refuses(web):
    clock = {"now": 1000.0}
    limiter = web._RateLimiter(count=3, window=60.0, clock=lambda: clock["now"])
    assert [limiter.check("example.com") for _ in range(3)] == [None, None, None]

    wait = limiter.check("example.com")
    assert wait is not None and 0 < wait <= 60.0
    # A different host has its own budget.
    assert limiter.check("other.example") is None

    clock["now"] += 61.0
    assert limiter.check("example.com") is None


def test_the_limiter_is_per_host_not_global(web):
    limiter = web._RateLimiter(count=1, window=60.0, clock=lambda: 0.0)
    for host in ("a.example", "b.example", "c.example"):
        assert limiter.check(host) is None


def test_the_default_limits_are_the_documented_ones(web):
    assert web.RATE_LIMIT_COUNT == 5
    assert web.RATE_LIMIT_WINDOW_S == 60.0
    assert web.DEFAULT_TIMEOUT == 15.0
    assert web.MAX_TIMEOUT == 60.0
    assert web.MAX_REDIRECTS == 5


def test_a_timeout_is_clamped_to_the_maximum(web):
    assert web._clamp_timeout(5) == 5.0
    assert web._clamp_timeout(600) == 60.0
    assert web._clamp_timeout(0) == 15.0
    assert web._clamp_timeout(-1) == 15.0
    assert web._clamp_timeout("nonsense") == 15.0
    assert web._clamp_timeout(None) == 15.0


def test_the_fetcher_receives_the_timeout_and_the_cap(web, monkeypatch):
    seen = {}

    def spy(url, *, timeout=15.0, max_bytes=None, **kwargs):
        seen.update(url=url, timeout=timeout, max_bytes=max_bytes)
        return result(web, body=b"<p>x</p>")

    monkeypatch.setattr(web, "_FETCHER", spy)
    web.web_fetch("https://example.com/")
    assert seen["url"] == "https://example.com/"
    assert seen["timeout"] == 15.0
    assert seen["max_bytes"] == web.MAX_RESPONSE_BYTES


# --- the manifest ---------------------------------------------------------


def test_the_manifest_is_valid_json_and_declares_the_right_permissions():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["name"] == "web"
    assert manifest["name"].isidentifier()
    assert sorted(manifest["permissions"]) == ["net", "tools"]


def test_the_plugin_loads_through_the_host():
    from adaptive_harness.plugins.host import PluginHost

    host = PluginHost(project_root=".")
    host.discover()
    plugin = next(p for p in host.plugins if p.name == "web")
    assert plugin.ok, plugin.error
    assert plugin.permissions == frozenset({"tools", "net"})


def test_both_tools_are_registered_with_net_risk():
    from adaptive_harness.plugins.host import PluginHost

    host = PluginHost(project_root=".")
    host.discover()
    plugin = next(p for p in host.plugins if p.name == "web")
    tools = {tool.name: tool for tool in plugin.tools}
    assert {"web_fetch", "web_query"} <= set(tools), sorted(tools)

    for name, tool in tools.items():
        assert tool.risk == "net", f"{name} is {tool.risk}, which would misreport the risk"
        assert callable(tool.handler)
        assert tool.parameters["type"] == "object"
        assert tool.parameters["required"], f"{name} declares no required arguments"
        for argument, schema in tool.parameters["properties"].items():
            assert schema.get("type"), f"{name}.{argument} has no type"
            assert len(schema.get("description", "")) > 20, \
                f"{name}.{argument} has no description the model can act on"
        assert len(tool.description) > 40


def test_the_search_tool_is_not_named_after_the_builtin():
    """A plugin cannot shadow a built-in tool, and the host enforces it."""
    from adaptive_harness.plugins.host import RESERVED_TOOL_NAMES

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    names = {tool["name"] for tool in manifest["tools"]}
    assert not (names & RESERVED_TOOL_NAMES), sorted(names & RESERVED_TOOL_NAMES)


def test_every_declared_handler_exists_on_the_module(web):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for spec in manifest["tools"]:
        assert callable(getattr(web, spec["handler"], None)), spec["handler"]


def test_the_tools_run_through_the_host_adapter(tmp_path):
    """End to end: the plugin as the agent would see it, still no network."""
    from adaptive_harness.plugins.host import PluginHost

    host = PluginHost(project_root=tmp_path)
    host.discover()
    tools = {tool.name: tool for tool in host.build_tools(tmp_path)}
    assert "web_fetch" in tools

    refused = tools["web_fetch"].execute(url="file:///etc/passwd")
    assert refused.success is False
    assert "scheme" in refused.error
    assert "Traceback" not in refused.error

    listed = host.describe()
    assert "web 1.0.0" in listed
    assert "net" in listed


def test_the_plugin_needs_no_api_key_to_load():
    """The key is read at call time, so installing the plugin never needs one."""
    assert (BUNDLED_DIR / "plugin.py").exists()
    module = _load_module()
    assert module._search_api_key() == ""
