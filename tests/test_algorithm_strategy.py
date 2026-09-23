"""Tests for AlgorithmStrategy: sort, binary search, and shortest path."""

import pytest
from adaptive_harness.models.domain import Result, Task
from adaptive_harness.strategies.algorithms import AlgorithmStrategy


@pytest.fixture
def strategy():
    return AlgorithmStrategy()


def test_sort_algorithm(strategy):
    t = Task(text="sort: 9 3 2 1 7")
    r = strategy.run_with_timing(t)
    assert r.success
    assert r.value == [1, 2, 3, 7, 9]
    v = strategy.verify(t, r)
    assert v.success


def test_sort_with_commas_and_floats(strategy):
    t = Task(text="sort this list: 4.5, 1.2, 9.8, 3.1")
    r = strategy.run_with_timing(t)
    assert r.success
    assert r.value == [1.2, 3.1, 4.5, 9.8]
    assert strategy.verify(t, r).success


def test_sort_verification_rejects_unsorted(strategy):
    t = Task(text="sort: 5 2 8 1")
    bad_result = Result(
        value=[1, 5, 2, 8],  # not sorted
        strategy_name="algorithms",
        success=True,
        metadata={"operation": "sort", "original": [5, 2, 8, 1], "reverse": False},
    )
    v = strategy.verify(t, bad_result)
    assert not v.success


def test_binary_search_found(strategy):
    t = Task(text="binary-search: target=8 data=1,3,5,8,10")
    r = strategy.run_with_timing(t)
    assert r.success
    assert r.value["found"] is True
    assert r.value["index"] == 3
    assert strategy.verify(t, r).success


def test_binary_search_not_found(strategy):
    t = Task(text="binary-search: target=7 data=1,3,5,8,10")
    r = strategy.run_with_timing(t)
    assert r.success
    assert r.value["found"] is False
    assert r.value["index"] == -1
    assert strategy.verify(t, r).success


def test_shortest_path_dijkstra(strategy):
    t = Task(text="shortest-path: start=A end=D edges=A-B:1,B-D:2,A-C:4,C-D:1")
    r = strategy.run_with_timing(t)
    assert r.success
    assert r.value["reachable"] is True
    assert r.value["path"] == ["A", "B", "D"]
    assert r.value["cost"] == 3.0
    v = strategy.verify(t, r)
    assert v.success


def test_shortest_path_verification_rejects_invalid_path(strategy):
    t = Task(text="shortest-path: start=A end=D edges=A-B:1,B-D:2,A-C:4,C-D:1")
    bad_result = Result(
        value={"path": ["A", "D"], "cost": 1.0, "reachable": True},  # No A-D edge
        strategy_name="algorithms",
        success=True,
        metadata={
            "operation": "shortest_path",
            "start": "A",
            "end": "D",
            "edges": [("A", "B", 1.0), ("B", "D", 2.0), ("A", "C", 4.0), ("C", "D", 1.0)],
            "cost": 1.0,
        },
    )
    v = strategy.verify(t, bad_result)
    assert not v.success
