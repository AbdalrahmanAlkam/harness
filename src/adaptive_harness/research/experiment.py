"""Empirical replication: seeded simulations with data hashes and 95% intervals.

An experiment is admissible evidence only when it is reproducible. The runner
pins NumPy/Python RNG seeds, executes the script in a subprocess, hashes the
data the script produced, and checks the declared prediction against the
observed statistic. A prediction that falls outside its 95% confidence
interval is reported as ``REJECTED_OUT_OF_INTERVAL`` — the experiment ran, but
it contradicted the theory, which is exactly the signal the convergence loop
needs in order to self-pivot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

EXPERIMENT_DIRNAME = "experiments"

# Artifacts a script may emit; each is hashed so results cannot be edited
# after the fact without invalidating the receipt.
_DATA_SUFFIXES = (".csv", ".json", ".npy", ".npz")


class ExperimentError(RuntimeError):
    """An experiment violated the reproducibility or evidence contract."""


def canonical_key(payload: Any) -> str:
    """Stable short digest of a payload, used to key memoized verdicts."""
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


@dataclass(frozen=True)
class Prediction:
    """A theory claim expressed as a testable numeric interval."""

    claim_id: str
    description: str
    predicted: float
    observed: float | None = None
    half_width_95: float | None = None
    relative: bool = True

    def evaluate(self) -> tuple[bool, str]:
        """Return ``(holds, detail)`` for the claim against the observed data."""
        if self.observed is None:
            return False, f"{self.claim_id}: no observed value was reported"
        if not (math.isfinite(self.predicted) and math.isfinite(self.observed)):
            return False, f"{self.claim_id}: non-finite prediction or observation"
        delta = abs(self.predicted - self.observed)
        if self.relative:
            scale = abs(self.predicted) if abs(self.predicted) > 1e-300 else 1.0
            ratio = delta / scale
            if self.half_width_95 is None:
                return ratio <= 1e-6, (
                    f"{self.claim_id}: relative error {ratio:.3g} "
                    f"(tolerance 1e-06, predicted {self.predicted:g}, observed {self.observed:g})")
            return ratio <= self.half_width_95, (
                f"{self.claim_id}: |predicted-observed|/predicted = {ratio:.4g} vs 95% "
                f"half-width {self.half_width_95:.4g} "
                f"(predicted {self.predicted:g}, observed {self.observed:g})")
        if self.half_width_95 is None:
            return delta <= 1e-6, (
                f"{self.claim_id}: absolute error {delta:.3g} "
                f"(predicted {self.predicted:g}, observed {self.observed:g})")
        return delta <= self.half_width_95, (
            f"{self.claim_id}: |predicted-observed| = {delta:.4g} vs 95% half-width "
            f"{self.half_width_95:.4g} (predicted {self.predicted:g}, observed {self.observed:g})")


@dataclass(frozen=True)
class ExperimentReceipt:
    """Immutable record that one experiment reproduced a claim."""

    experiment_id: str
    script: str
    seed: int
    status: str
    data_hashes: Mapping[str, str] = field(default_factory=dict)
    exit_code: int | None = None
    duration_ms: float = 0.0
    predictions: tuple[Mapping[str, Any], ...] = ()
    stdout: str = ""
    error: str | None = None

    @property
    def replicated(self) -> bool:
        return self.status == "REPLICATED"

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, "script": self.script, "seed": self.seed,
                "status": self.status, "data_hashes": dict(self.data_hashes),
                "exit_code": self.exit_code, "duration_ms": round(self.duration_ms, 2),
                "predictions": [dict(item) for item in self.predictions], "error": self.error}


def mean_confidence_interval_95(samples: Sequence[float]) -> tuple[float, float, float]:
    """Return ``(mean, half_width_95, n)`` for a sample, Student-t free.

    Uses the normal approximation with a small-sample guard: for ``n < 30`` the
    t-correction inflates the interval rather than understating uncertainty.
    """
    values = [float(value) for value in samples]
    n = len(values)
    if n == 0:
        raise ExperimentError("Cannot compute a confidence interval from zero samples")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 1
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    std_error = math.sqrt(variance / n)
    # Two-sided 95% t quantiles, tabulated for the sample sizes we support and
    # approximated by 1.96 beyond the table.
    t_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
               9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145,
               16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086, 22: 2.080,
               23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048,
               30: 2.045}
    t_critical = t_table.get(n, 1.96)
    return mean, t_critical * std_error, n


_SEED_PRELUDE = """
import random as _random
import numpy as _np
_random.seed({seed})
_np.random.seed({seed})
"""


@dataclass
class ExperimentRunner:
    """Execute seeded experiment scripts and adjudicate their predictions.

    Receipts are memoized on ``(script digest, seed)``. The data files are
    already on disk and hashed, so re-executing an unchanged script would
    reproduce byte-identical output at full cost.
    """

    experiment_dir: Path
    timeout_s: float = 300.0
    python_executable: str = field(default=sys.executable)
    receipts: list[ExperimentReceipt] = field(default_factory=list)
    _cache: dict[tuple[str, int], ExperimentReceipt] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.experiment_dir = Path(self.experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def scripts(self) -> list[Path]:
        return sorted(path for path in self.experiment_dir.glob("*.py")
                      if not path.name.startswith("."))

    def _hash_artifacts(self, before: Mapping[str, float]) -> dict[str, str]:
        """Hash data files, recording only those the script created or touched."""
        hashes: dict[str, str] = {}
        for path in sorted(self.experiment_dir.iterdir()):
            if path.suffix not in _DATA_SUFFIXES or not path.is_file():
                continue
            previous = before.get(path.name)
            current = path.stat().st_mtime_ns
            if previous is not None and abs(current - previous) < 1e-9:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[path.name] = f"sha256:{digest}"
        return hashes

    def _snapshot(self) -> dict[str, float]:
        return {path.name: float(path.stat().st_mtime_ns)
                for path in self.experiment_dir.iterdir() if path.is_file()}

    def run_script(self, script: str | Path, *, seed: int, experiment_id: str | None = None,
                   predictions: Sequence[Prediction] = (), use_cache: bool = True) -> ExperimentReceipt:
        """Run one experiment script under a pinned seed and adjudicate claims."""
        import hashlib
        import time

        path = Path(script)
        if not path.is_absolute():
            path = self.experiment_dir / path
        if not path.is_file():
            return ExperimentReceipt(experiment_id or path.stem, str(path), seed, "MISSING",
                                    None, 0.0, (), "experiment script does not exist")
        source = path.read_text(encoding="utf-8", errors="replace")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        identifier = experiment_id or path.stem

        # Predictions are part of the verdict, so a different claim set must
        # re-adjudicate even when the script is unchanged.
        claim_key = canonical_key([asdict(item) for item in predictions])
        cache_key = (f"{digest}:{claim_key}", int(seed))
        cached = self._cache.get(cache_key) if use_cache else None
        if cached is not None and self._artifacts_intact(cached):
            return cached

        guarded = self.experiment_dir / f".{path.stem}.seeded.py"
        guarded.write_text(_SEED_PRELUDE.format(seed=seed) + source, encoding="utf-8")
        before = self._snapshot()
        started = time.perf_counter()
        try:
            completed = subprocess.run([self.python_executable, str(guarded)], capture_output=True,
                                       text=True, timeout=self.timeout_s, check=False,
                                       cwd=str(self.experiment_dir))
            exit_code: int | None = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired:
            exit_code, stdout, stderr = None, "", f"timeout after {self.timeout_s:g}s"
        except OSError as exc:
            exit_code, stdout, stderr = None, "", f"{type(exc).__name__}: {exc}"
        finally:
            guarded.unlink(missing_ok=True)
        duration_ms = (time.perf_counter() - started) * 1000.0
        hashes = self._hash_artifacts(before)

        if exit_code != 0:
            status = "REJECTED_NONZERO" if exit_code is not None else "REJECTED_TIMEOUT"
            receipt = ExperimentReceipt(identifier, str(path), seed, status, hashes,
                                        exit_code, duration_ms, (), (stdout + stderr)[-2000:],
                                        f"experiment exited with code {exit_code}")
            self._cache[cache_key] = receipt
            return receipt
        if not hashes:
            receipt = ExperimentReceipt(identifier, str(path), seed, "REJECTED_NO_DATA", {},
                                        exit_code, duration_ms, (), (stdout + stderr)[-2000:],
                                        "experiment produced no hashed data artifact")
            self._cache[cache_key] = receipt
            return receipt

        evaluated: list[Mapping[str, Any]] = []
        all_hold = True
        for prediction in predictions:
            holds, detail = prediction.evaluate()
            all_hold = all_hold and holds
            evaluated.append({**asdict(prediction), "holds": holds, "detail": detail})
        status = "REPLICATED" if all_hold else "REJECTED_OUT_OF_INTERVAL"
        receipt = ExperimentReceipt(identifier, str(path), seed, status, hashes,
                                    exit_code, duration_ms, tuple(evaluated), (stdout + stderr)[-2000:],
                                    None if all_hold else "; ".join(
                                        item["detail"] for item in evaluated if not item["holds"]))
        self._cache[cache_key] = receipt
        return receipt

    def _artifacts_intact(self, receipt: ExperimentReceipt) -> bool:
        """A cached receipt is only reusable while its data files still hash the same."""
        import hashlib
        for name, digest in receipt.data_hashes.items():
            path = self.experiment_dir / name
            if not path.is_file():
                return False
            if f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}" != digest:
                return False
        return True

    def run_all(self, seeds: Mapping[str, int] | None = None,
                predictions: Mapping[str, Sequence[Prediction]] | None = None) -> list[ExperimentReceipt]:
        """Run every experiment script with its pinned seed."""
        seed_map = dict(seeds or {})
        claim_map = dict(predictions or {})
        receipts = [
            self.run_script(path, seed=seed_map.get(path.name, 1234),
                            experiment_id=path.stem,
                            predictions=claim_map.get(path.name, ()))
            for path in self.scripts()
        ]
        self.receipts = receipts
        return receipts

    def record(self, receipt: ExperimentReceipt, ledger: Any = None) -> ExperimentReceipt:
        self.receipts.append(receipt)
        index_path = self.experiment_dir.parent / "experiment_receipts.json"
        existing: list[dict[str, Any]] = []
        if index_path.is_file():
            try:
                existing = json.loads(index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = []
        existing = [item for item in existing if item.get("script") != receipt.script]
        existing.append(receipt.to_dict())
        index_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        if ledger is not None:
            ledger.append("EXPERIMENT_VERIFIED" if receipt.replicated else "EXPERIMENT_REJECTED",
                          {"agent_id": "experiment_runner", "role": "Experiment Runner"},
                          {"agent_id": "empirical_lead", "role": "Empirical Lead"},
                          {"experiment_id": receipt.experiment_id,
                           "script": str(Path(receipt.script).name),
                           "seed": receipt.seed, "status": receipt.status,
                           "data_hashes": dict(receipt.data_hashes),
                           "error": receipt.error})
        return receipt

    @property
    def all_replicated(self) -> bool:
        return bool(self.receipts) and all(receipt.replicated for receipt in self.receipts)
