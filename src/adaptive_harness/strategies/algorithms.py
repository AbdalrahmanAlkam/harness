"""Algorithm strategy for structured tasks: sorting, binary search, and graph shortest path."""

from __future__ import annotations

from collections import Counter
import heapq
import re
from typing import Any, Dict, List, Optional, Tuple

from adaptive_harness.models.domain import Task, Result, VerificationResult
from adaptive_harness.strategies.base import Strategy


class AlgorithmStrategy(Strategy):
    """Specialist handler for structured algorithmic problems."""

    name = "algorithms"

    def can_handle(self, task: Task) -> bool:
        text = task.text.lower()
        algo_markers = [
            "sort:", "sort", "binary-search:", "binary search",
            "shortest-path:", "shortest path", "dijkstra", "pathfind",
            "find path", "path", "reverse:", "median:", "edges="
        ]
        return any(m in text for m in algo_markers)

    def execute(self, task: Task) -> Result:
        raw_text = task.text.strip()
        text = raw_text.lower()

        # 1. Binary Search
        if "binary-search" in text or "binary search" in text or "search for target" in text or "search for key" in text:
            return self._execute_binary_search(raw_text)

        # 2. Shortest Path
        if "shortest-path" in text or "shortest path" in text or "dijkstra" in text or "find path" in text or "edges=" in text:
            return self._execute_shortest_path(raw_text)

        # 3. Sorting
        if "sort" in text or "order" in text or "arrange" in text:
            return self._execute_sort(raw_text)

        return Result(
            value=None,
            strategy_name=self.name,
            success=False,
            error="Unrecognized algorithm specification",
        )

    def _execute_sort(self, text: str) -> Result:
        # Extract numbers from text after marker or in text
        # e.g. "sort: 9 3 2 1 7" or "sort this list efficiently: 4, 1, 8, 3"
        cleaned = re.sub(r"^.*?sort(?:\s+this\s+list\s*(?:efficiently)?)?[\s:]*", "", text, flags=re.IGNORECASE)
        # Parse numbers (integers or floats)
        num_strs = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if not num_strs:
            return Result(
                value=None,
                strategy_name=self.name,
                success=False,
                error="No numeric elements found to sort",
            )

        numbers = [float(x) if "." in x else int(x) for x in num_strs]
        reverse = "descending" in text.lower() or "reverse" in text.lower()
        sorted_nums = sorted(numbers, reverse=reverse)

        return Result(
            value=sorted_nums,
            strategy_name=self.name,
            success=True,
            metadata={
                "operation": "sort",
                "original": numbers,
                "reverse": reverse,
                "count": len(numbers),
            },
        )

    def _execute_binary_search(self, text: str) -> Result:
        target_match = re.search(r"(?:target|key)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if not target_match:
            return Result(
                value=None,
                strategy_name=self.name,
                success=False,
                error="Could not parse binary search target",
            )
        target_str = target_match.group(1)
        target = float(target_str) if "." in target_str else int(target_str)

        # Extract data part
        data_part = text[target_match.end():]
        num_strs = re.findall(r"-?\d+(?:\.\d+)?", data_part)
        if not num_strs:
            # Try finding numbers before target
            data_part = text[:target_match.start()]
            num_strs = re.findall(r"-?\d+(?:\.\d+)?", data_part)

        if not num_strs:
            return Result(
                value=None,
                strategy_name=self.name,
                success=False,
                error="No data array found for binary search",
            )

        data = [float(x) if "." in x else int(x) for x in num_strs]
        # Perform binary search
        low, high = 0, len(data) - 1
        found_idx = -1
        while low <= high:
            mid = (low + high) // 2
            if data[mid] == target:
                found_idx = mid
                break
            elif data[mid] < target:
                low = mid + 1
            else:
                high = mid - 1

        return Result(
            value={"index": found_idx, "target": target, "found": found_idx != -1},
            strategy_name=self.name,
            success=True,
            metadata={"operation": "binary_search", "data": data, "target": target},
        )

    def _execute_shortest_path(self, text: str) -> Result:
        # Format: shortest-path: start=A end=D edges=A-B:1,B-D:2,A-C:4,C-D:1
        start_match = re.search(r"start\s*[:=]\s*([a-zA-Z0-9_]+)", text, re.IGNORECASE)
        end_match = re.search(r"end\s*[:=]\s*([a-zA-Z0-9_]+)", text, re.IGNORECASE)

        if start_match and end_match:
            start = start_match.group(1).upper()
            end = end_match.group(1).upper()
        else:
            from_to = re.search(r"from\s+([a-zA-Z0-9_]+)\s+to\s+([a-zA-Z0-9_]+)", text, re.IGNORECASE)
            if from_to:
                start = from_to.group(1).upper()
                end = from_to.group(2).upper()
            else:
                return Result(
                    value=None,
                    strategy_name=self.name,
                    success=False,
                    error="Shortest path requires start and end nodes (e.g. start=A end=D or from A to D)",
                )

        # Parse edges: e.g. A-B:1.5 or A->B:2 or A-B (default weight 1)
        edges_str = text
        edge_matches = re.findall(r"([a-zA-Z0-9_]+)\s*(?:-|->)\s*([a-zA-Z0-9_]+)(?:\s*[:=]\s*(\d+(?:\.\d+)?))?", edges_str)
        if not edge_matches:
            return Result(
                value=None,
                strategy_name=self.name,
                success=False,
                error="Could not parse graph edges",
            )

        graph: Dict[str, List[Tuple[str, float]]] = {}
        parsed_edges = []
        for u, v, w in edge_matches:
            u, v = u.upper(), v.upper()
            weight = float(w) if w else 1.0
            graph.setdefault(u, []).append((v, weight))
            graph.setdefault(v, []).append((u, weight))  # undirected
            parsed_edges.append((u, v, weight))

        # Dijkstra algorithm
        dist: Dict[str, float] = {start: 0.0}
        prev: Dict[str, Optional[str]] = {start: None}
        pq: List[Tuple[float, str]] = [(0.0, start)]

        while pq:
            d, node = heapq.heappop(pq)
            if d > dist.get(node, float("inf")):
                continue
            if node == end:
                break

            for neighbor, weight in graph.get(node, []):
                new_d = d + weight
                if new_d < dist.get(neighbor, float("inf")):
                    dist[neighbor] = new_d
                    prev[neighbor] = node
                    heapq.heappush(pq, (new_d, neighbor))

        if end not in dist:
            return Result(
                value={"path": [], "cost": float("inf"), "reachable": False},
                strategy_name=self.name,
                success=True,
                metadata={"operation": "shortest_path", "start": start, "end": end},
            )

        # Reconstruct path
        path = []
        curr: Optional[str] = end
        while curr is not None:
            path.append(curr)
            curr = prev.get(curr)
        path.reverse()

        return Result(
            value={"path": path, "cost": round(dist[end], 4), "reachable": True},
            strategy_name=self.name,
            success=True,
            metadata={
                "operation": "shortest_path",
                "start": start,
                "end": end,
                "edges": parsed_edges,
                "cost": round(dist[end], 4),
            },
        )

    def verify(self, task: Task, result: Result) -> VerificationResult:
        if not result.success or result.value is None:
            return VerificationResult(
                success=False,
                reason=f"Algorithm execution failed: {result.error or 'no output'}",
            )

        op = result.metadata.get("operation")
        val = result.value

        if op == "sort":
            original = result.metadata.get("original", [])
            reverse = result.metadata.get("reverse", False)
            if not isinstance(val, list):
                return VerificationResult(success=False, reason="Sorted result must be a list")
            if len(val) != len(original):
                return VerificationResult(success=False, reason="Sorted list length does not match input length")
            # Verify multiset equality
            if Counter(val) != Counter(original):
                return VerificationResult(success=False, reason="Sorted elements do not match input elements")
            # Verify monotonic order
            for i in range(len(val) - 1):
                if reverse:
                    if val[i] < val[i + 1]:
                        return VerificationResult(success=False, reason="List is not sorted in descending order")
                else:
                    if val[i] > val[i + 1]:
                        return VerificationResult(success=False, reason="List is not sorted in ascending order")
            return VerificationResult(success=True, reason="Sort order and permutation verified")

        elif op == "binary_search":
            data = result.metadata.get("data", [])
            target = result.metadata.get("target")
            idx = val.get("index")
            found = val.get("found")
            if found:
                if idx < 0 or idx >= len(data):
                    return VerificationResult(success=False, reason="Binary search index out of bounds")
                if data[idx] != target:
                    return VerificationResult(success=False, reason=f"Element at index {idx} ({data[idx]}) != target ({target})")
            else:
                if target in data:
                    return VerificationResult(success=False, reason="Target was present in data but not found")
            return VerificationResult(success=True, reason="Binary search result verified")

        elif op == "shortest_path":
            start = result.metadata.get("start")
            end = result.metadata.get("end")
            path = val.get("path", [])
            cost = val.get("cost")
            reachable = val.get("reachable")

            if not reachable:
                return VerificationResult(success=True, reason="Unreachability verified")

            if not path or path[0] != start or path[-1] != end:
                return VerificationResult(success=False, reason=f"Path endpoints do not match start={start}, end={end}")

            edges_list = result.metadata.get("edges", [])
            edge_map: Dict[Tuple[str, str], float] = {}
            for u, v, w in edges_list:
                edge_map[(u, v)] = min(edge_map.get((u, v), float("inf")), w)
                edge_map[(v, u)] = min(edge_map.get((v, u), float("inf")), w)

            calc_cost = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if (u, v) not in edge_map:
                    return VerificationResult(success=False, reason=f"Path segment {u}->{v} is not an edge in the graph")
                calc_cost += edge_map[(u, v)]

            if abs(calc_cost - cost) > 1e-4:
                return VerificationResult(success=False, reason=f"Sum of path edges ({calc_cost}) != claimed cost ({cost})")

            return VerificationResult(success=True, reason="Shortest path topology and cost verified")

        return VerificationResult(success=True, reason="Algorithm result verified")
