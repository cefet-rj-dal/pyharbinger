"""Online / streaming layer for Harbinger detectors.

Python port of the R online framework (``har_online_memory``,
``har_online_executor``, ``har_online_sources``, ``har_online_session``). The
layer is additive: it wraps any existing offline detector and drives it over a
stream, handling ingestion, batching, memory policies, execution strategies, and
a trace that yields Detection Probability (DP) and Detection Lag (DL).

The pieces:

* **Sources** -- :class:`HarSourceSimulated`, :class:`HarSourceDataframe`,
  :class:`HarSourceCallback` produce observations in pull mode. (The R Kafka
  stub is intentionally not ported.)
* **Memory policies** -- :class:`HarMemoryFull`, :class:`HarMemorySliding`,
  :class:`HarMemoryLastObservations` bound the in-memory context.
* **Execution strategies** -- :class:`HarOnlineRefitFull`,
  :class:`HarOnlineDetectOnly`, :class:`HarOnlineIncremental` decide how the
  detector is invoked per batch.
* **Session** -- :class:`HarOnlineSession` orchestrates everything.

Positions are 0-based, matching the rest of :mod:`pyharbinger` (the R original
is 1-based), so a stream position maps to the same ``idx`` an offline detection
would report.

References
----------
- Ogasawara E, Salles R, Porto F, Pacitti E (2025). Event Detection in Time
  Series. Springer. doi:10.1007/978-3-031-75941-3

Porting
-------
Mirrors ``har_online_*.R``. Fidelity: exact on the detection/trace outputs
(idx shifted to 0-based); timing columns are environment-dependent.
"""
from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import pandas as pd

__all__ = [
    "HarMemoryFull",
    "HarMemoryLastObservations",
    "HarMemorySliding",
    "HarOnlineDetectOnly",
    "HarOnlineIncremental",
    "HarOnlineRefitFull",
    "HarOnlineSession",
    "HarSourceCallback",
    "HarSourceDataframe",
    "HarSourceSimulated",
    "har_memory_label",
]


# --- memory policies ---------------------------------------------------------

class HarMemoryFull:
    """Keep every observation seen by the session."""

    mode = "full"


class HarMemorySliding:
    """Keep only the most recent ``batches`` completed batches."""

    mode = "sliding_batches"

    def __init__(self, batches: int):
        batches = int(batches)
        if batches < 1:
            raise ValueError("batches must be a positive integer.")
        self.batches = batches


class HarMemoryLastObservations:
    """Keep only the most recent ``n`` observations."""

    mode = "last_observations"

    def __init__(self, n: int):
        n = int(n)
        if n < 1:
            raise ValueError("n must be a positive integer.")
        self.n = n


def har_memory_label(memory) -> str:
    """Short label for a memory policy (mirrors R ``har_memory_label``)."""
    if isinstance(memory, HarMemoryFull):
        return "full"
    if isinstance(memory, HarMemorySliding):
        return f"sliding_batches_{memory.batches}"
    if isinstance(memory, HarMemoryLastObservations):
        return f"last_observations_{memory.n}"
    return "custom"


# --- execution strategies ----------------------------------------------------

class HarOnlineRefitFull:
    """Re-fit the detector on the in-memory series before every batch."""

    mode = "refit_full"

    def __init__(self, fit_on_warmup: bool = True):
        self.fit_on_warmup = bool(fit_on_warmup)
        self.fit_each_run = True


class HarOnlineDetectOnly:
    """Only call ``detect`` during the loop (fit at most once on warm-up)."""

    mode = "detect_only"

    def __init__(self, fit_on_warmup: bool = False):
        self.fit_on_warmup = bool(fit_on_warmup)
        self.fit_each_run = False


class HarOnlineIncremental:
    """Call the detector's ``online_update`` when present, else detect-only."""

    mode = "incremental"

    def __init__(self, fit_on_warmup: bool = True):
        self.fit_on_warmup = bool(fit_on_warmup)
        self.fit_each_run = False


# --- observation normalization ----------------------------------------------

def _normalize_observation(observation, fallback_idx: int) -> dict:
    """Coerce a raw observation into the ``{idx, value, timestamp, payload}`` contract.

    Mirrors R ``har_normalize_source_observation``: a scalar or a one-row
    DataFrame becomes a value payload with the fallback index; a mapping must
    carry a ``value`` and has its missing fields filled in.
    """
    if np.isscalar(observation):
        return {"idx": fallback_idx, "value": observation, "timestamp": np.nan, "payload": observation}
    if isinstance(observation, pd.DataFrame) and len(observation) == 1:
        return {"idx": fallback_idx, "value": observation, "timestamp": np.nan, "payload": observation}
    if not isinstance(observation, dict):
        raise TypeError("Observation must be scalar, one-row DataFrame, or dict.")
    if "value" not in observation or observation["value"] is None:
        raise ValueError("Observation dict must include a value field.")
    obs = dict(observation)
    obs.setdefault("idx", fallback_idx)
    obs.setdefault("timestamp", np.nan)
    obs.setdefault("payload", obs["value"])
    return obs


# --- sources -----------------------------------------------------------------

class HarSourceSimulated:
    """Replay source over an in-memory vector, matrix, or DataFrame.

    Mirrors R ``har_source_simulated``. Observations are produced in time order;
    ``next_observation`` returns a normalized observation dict or ``None`` when
    exhausted.
    """

    def __init__(self, data, timestamp=None, name: str = "simulated"):
        if data is None:
            raise ValueError("data must be provided.")
        self.name = name
        self._is_frame = isinstance(data, pd.DataFrame)
        self.data = data
        self.n = len(data)
        if timestamp is not None and len(timestamp) != self.n:
            raise ValueError("timestamp must have the same length as data.")
        self.timestamp = timestamp
        self.cursor = 0

    def next_observation(self):
        if self.cursor >= self.n:
            return None
        i = self.cursor
        self.cursor += 1
        value = self.data.iloc[[i]] if self._is_frame else self.data[i]
        ts = np.nan if self.timestamp is None else self.timestamp[i]
        return {"idx": i, "value": value, "timestamp": ts, "payload": value}

    def info(self) -> dict:
        return {"name": self.name, "type": "simulated", "n": self.n, "cursor": self.cursor}


class HarSourceDataframe(HarSourceSimulated):
    """Replay source for a DataFrame, selecting value and optional timestamp columns.

    Mirrors R ``har_source_dataframe``.
    """

    def __init__(self, data, timestamp_col=None, value_cols=None, name: str = "dataframe"):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a DataFrame.")
        ts = None
        ts_name = None
        if timestamp_col is not None:
            ts_name = timestamp_col if isinstance(timestamp_col, str) else data.columns[int(timestamp_col)]
            if ts_name not in data.columns:
                raise ValueError("timestamp_col does not reference a valid column.")
            ts = data[ts_name].to_numpy()
        if value_cols is None:
            value_names = [c for c in data.columns if c != ts_name]
        elif isinstance(value_cols, str):
            value_names = [value_cols]
        elif all(isinstance(c, str) for c in value_cols):
            value_names = list(value_cols)
        else:
            value_names = [data.columns[int(i)] for i in value_cols]
        if not value_names or any(c not in data.columns for c in value_names):
            raise ValueError("value_cols must reference at least one valid column.")
        values = data[value_names].reset_index(drop=True)
        # A single value column replays as a scalar series, matching R's behavior.
        payload = values.iloc[:, 0].to_numpy() if len(value_names) == 1 else values
        super().__init__(payload, timestamp=ts, name=name)
        self.timestamp_col = ts_name
        self.value_cols = value_names

    def info(self) -> dict:
        base = super().info()
        base.update({"type": "dataframe", "timestamp_col": self.timestamp_col, "value_cols": self.value_cols})
        return base


class HarSourceCallback:
    """Pull source that calls ``poll_fn`` for each observation (``None`` ends the stream).

    Mirrors R ``har_source_callback``.
    """

    def __init__(self, poll_fn, name: str = "callback"):
        if not callable(poll_fn):
            raise TypeError("poll_fn must be callable.")
        self.name = name
        self.poll_fn = poll_fn
        self.cursor = 0

    def next_observation(self):
        observation = self.poll_fn()
        if observation is None:
            return None
        obs = _normalize_observation(observation, self.cursor)
        self.cursor += 1
        return obs

    def info(self) -> dict:
        return {"name": self.name, "type": "callback", "cursor": self.cursor}


# --- session -----------------------------------------------------------------

class HarOnlineSession:
    """Streaming session that drives an offline Harbinger detector over a source.

    Parameters mirror R ``har_online_session``: a ``source`` (or ``None`` for
    push-only), the ``detector``, an ``executor`` strategy, ``warmup_size``,
    ``batch_size``, a ``memory`` policy, and a ``mode`` (``"auto"``/``"pull"``/
    ``"push"``).
    """

    def __init__(
        self,
        source,
        detector,
        executor=None,
        warmup_size: int = 30,
        batch_size: int = 30,
        memory=None,
        mode: str = "auto",
    ):
        if detector is None:
            raise ValueError("detector must be provided.")
        if mode not in ("auto", "pull", "push"):
            raise ValueError("mode must be 'auto', 'pull', or 'push'.")
        warmup_size = int(warmup_size)
        batch_size = int(batch_size)
        if warmup_size < 0:
            raise ValueError("warmup_size must be a non-negative integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        self.source = source
        self.detector = detector
        self.executor = executor if executor is not None else HarOnlineRefitFull()
        self.warmup_size = warmup_size
        self.batch_size = batch_size
        self.memory = memory if memory is not None else HarMemoryFull()
        # "auto" behaves as pull with a source, push without one.
        self.mode = "push" if (mode == "auto" and source is None) else ("pull" if mode == "auto" else mode)

        self.queue: list = []
        self.pull_exhausted = False
        self.fitted = False
        self.closed = False

        self.values: list = []
        self.positions: list[int] = []
        self.timestamps: list = []
        self.batch_seen: list[int] = []
        self.global_count = -1  # highest idx assigned so far (0-based); -1 = none

        self.warmup_consumed = 0
        self.pending_since_last_run = 0
        self.run_count = 0

        self.trace_map: OrderedDict[int, dict] = OrderedDict()
        self.batch_log = pd.DataFrame(
            {c: pd.Series(dtype=t) for c, t in {
                "batch_id": int, "memory_size": int, "fit_time_sec": float,
                "detect_time_sec": float, "total_time_sec": float,
            }.items()}
        )

    # -- ingestion -----------------------------------------------------------

    def ingest(self, observation) -> HarOnlineSession:
        """Push one observation, a list of observations, or a DataFrame into the queue."""
        for obs in _as_observation_list(observation):
            fallback = self.global_count + 1 + len(self.queue)
            self.queue.append(_normalize_observation(obs, fallback))
        return self

    # -- lifecycle -----------------------------------------------------------

    def fit(self, **kwargs) -> HarOnlineSession:
        """Consume warm-up observations and optionally fit the detector once."""
        while self.warmup_consumed < self.warmup_size:
            obs = self._fetch_one()
            if obs is None:
                break
            self._store_observation(obs)
            self.warmup_consumed += 1

        if getattr(self.executor, "fit_on_warmup", False) and self.values:
            self.detector = self.detector.fit(self._series(), **kwargs)
            self.fitted = True
        return self

    def step(self, **kwargs) -> HarOnlineSession:
        """Consume at most one observation and run a cycle when the batch is full."""
        if self.closed:
            return self
        if self.warmup_consumed < self.warmup_size:
            return self.fit(**kwargs)

        obs = self._fetch_one()
        if obs is None:
            if self.pending_since_last_run > 0:
                self._execute_cycle(**kwargs)
                self.pending_since_last_run = 0
            self.closed = self.is_finished()
            return self

        self._store_observation(obs)
        self.pending_since_last_run += 1
        if self.pending_since_last_run >= self.batch_size:
            self._execute_cycle(**kwargs)
            self.pending_since_last_run = 0

        self.closed = self.is_finished()
        return self

    def run(self, max_batches=None, **kwargs) -> HarOnlineSession:
        """Step until the source is exhausted and the queue empty (or ``max_batches``)."""
        if max_batches is not None:
            max_batches = int(max_batches)
            if max_batches < 1:
                raise ValueError("max_batches must be a positive integer.")
        while not self.is_finished():
            before = self.run_count
            self.step(**kwargs)
            if max_batches is not None and self.run_count >= max_batches:
                break
            if self.run_count == before and self.is_finished():
                break
        return self

    def is_finished(self) -> bool:
        """True when the queue is empty (push) or also the pull source is exhausted."""
        queue_empty = len(self.queue) == 0
        if self.mode == "push":
            return queue_empty
        return queue_empty and self.pull_exhausted

    # -- collectors ----------------------------------------------------------

    def collect_trace(self) -> pd.DataFrame:
        """One row per observation with the DP/DL quantities."""
        cols = [
            "idx", "timestamp", "batch_id_first_seen", "batch_frequency",
            "detection_frequency", "first_detected_batch", "last_detected_batch",
            "detection_probability", "detection_lag_batches",
            "detection_lag_observations", "event_type",
        ]
        if not self.trace_map:
            return pd.DataFrame({c: [] for c in cols})
        rows = [self._trace_entry_to_row(self.trace_map[k]) for k in sorted(self.trace_map)]
        return pd.DataFrame(rows, columns=cols)

    def collect_detection(self) -> pd.DataFrame:
        """Materialize the final detection table with DP/DL summary columns."""
        trace = self.collect_trace()
        if len(trace) == 0:
            return pd.DataFrame({"idx": [], "event": [], "type": []})
        detected = trace["detection_frequency"] > 0
        return pd.DataFrame({
            "idx": trace["idx"].to_numpy(),
            "event": detected.to_numpy(),
            "type": np.where(detected, trace["event_type"], "").astype(object),
            "detection_probability": trace["detection_probability"].to_numpy(),
            "detection_lag_batches": trace["detection_lag_batches"].to_numpy(),
            "detection_lag_observations": trace["detection_lag_observations"].to_numpy(),
        })

    def collect_batch_log(self) -> pd.DataFrame:
        """One row per completed batch (sizes and timings)."""
        return self.batch_log

    # -- internals -----------------------------------------------------------

    def _fetch_one(self):
        if self.queue:
            return self.queue.pop(0)
        if self.mode == "push":
            return None
        if self.source is None:
            self.pull_exhausted = True
            return None
        obs = self.source.next_observation()
        if obs is None:
            self.pull_exhausted = True
        return obs

    def _store_observation(self, observation):
        observation = _normalize_observation(observation, self.global_count + 1)
        idx = int(observation["idx"])
        self.global_count = max(self.global_count, idx)
        self.values.append(observation["value"])
        self.positions.append(idx)
        self.timestamps.append(observation["timestamp"])
        self.batch_seen.append(self.run_count + 1)
        if idx not in self.trace_map:
            self.trace_map[idx] = {
                "idx": idx,
                "timestamp": observation["timestamp"],
                "batch_id_first_seen": self.run_count + 1,
                "batch_ids_present": [],
                "detected_in_batches": [],
                "first_detected_batch": None,
                "last_detected_batch": None,
                "event_type": "event",
            }

    def _series(self):
        if not self.values:
            return np.array([])
        first = self.values[0]
        if isinstance(first, pd.DataFrame):
            return pd.concat(self.values, ignore_index=True)
        return np.asarray([float(v) for v in self.values])

    def _execute_cycle(self, **kwargs):
        if not self.values:
            return
        batch_id = self.run_count + 1
        for pos in self.positions:
            present = self.trace_map[pos]["batch_ids_present"]
            if batch_id not in present:
                present.append(batch_id)

        serie = self._series()
        fit_time = 0.0
        detect_time = 0.0

        if self.executor.fit_each_run or not self.fitted and getattr(self.executor, "fit_on_warmup", False):
            t0 = time.perf_counter()
            self.detector = self.detector.fit(serie, **kwargs)
            fit_time = time.perf_counter() - t0
            self.fitted = True

        if self.executor.mode == "incremental" and hasattr(self.detector, "online_update"):
            self.detector = self.detector.online_update(serie, **kwargs)

        t0 = time.perf_counter()
        detection = self.detector.detect(serie, **kwargs)
        detect_time = time.perf_counter() - t0

        self._update_trace_from_detection(detection, batch_id)
        self.run_count = batch_id
        self.batch_log.loc[len(self.batch_log)] = {
            "batch_id": batch_id,
            "memory_size": len(self.values),
            "fit_time_sec": fit_time,
            "detect_time_sec": detect_time,
            "total_time_sec": fit_time + detect_time,
        }
        self._apply_memory()

    def _update_trace_from_detection(self, detection, batch_id):
        if detection is None or len(detection) == 0:
            return
        if not {"idx", "event"}.issubset(detection.columns):
            raise ValueError("Wrapped detector returned an invalid detection object.")
        event = detection["event"].fillna(False).to_numpy(dtype=bool)
        types = (
            detection["type"].fillna("").to_numpy(dtype=object)
            if "type" in detection else np.full(len(event), "", dtype=object)
        )
        for i in range(min(len(detection), len(self.positions))):
            if not event[i]:
                continue
            entry = self.trace_map[self.positions[i]]
            if batch_id not in entry["detected_in_batches"]:
                entry["detected_in_batches"].append(batch_id)
            if entry["first_detected_batch"] is None:
                entry["first_detected_batch"] = batch_id
            entry["last_detected_batch"] = batch_id
            if types[i]:
                entry["event_type"] = types[i]
            elif not entry["event_type"]:
                entry["event_type"] = "event"

    def _apply_memory(self):
        if isinstance(self.memory, HarMemoryFull):
            return
        keep = list(range(len(self.values)))
        if isinstance(self.memory, HarMemoryLastObservations):
            keep = keep[-min(len(self.values), self.memory.n):]
        elif isinstance(self.memory, HarMemorySliding):
            keep_batches = sorted(set(self.batch_seen))[-self.memory.batches:]
            keep = [i for i, b in enumerate(self.batch_seen) if b in keep_batches]
        self.values = [self.values[i] for i in keep]
        self.positions = [self.positions[i] for i in keep]
        self.timestamps = [self.timestamps[i] for i in keep]
        self.batch_seen = [self.batch_seen[i] for i in keep]

    def _trace_entry_to_row(self, entry) -> dict:
        bf = len(set(entry["batch_ids_present"]))
        df = len(set(entry["detected_in_batches"]))
        prob = 0.0 if bf == 0 else df / bf
        first = entry["first_detected_batch"]
        lag_batches = None if first is None else first - entry["batch_id_first_seen"]
        lag_obs = None if lag_batches is None else (lag_batches + 1) * self.batch_size
        return {
            "idx": entry["idx"],
            "timestamp": entry["timestamp"],
            "batch_id_first_seen": entry["batch_id_first_seen"],
            "batch_frequency": bf,
            "detection_frequency": df,
            "first_detected_batch": first,
            "last_detected_batch": entry["last_detected_batch"],
            "detection_probability": prob,
            "detection_lag_batches": lag_batches,
            "detection_lag_observations": lag_obs,
            "event_type": entry["event_type"],
        }


def _as_observation_list(observation) -> list:
    """Split a batch ingestion argument into individual observations (R ``har_as_observation_list``)."""
    if isinstance(observation, pd.DataFrame):
        return [observation.iloc[[i]] for i in range(len(observation))]
    if isinstance(observation, dict) and observation.get("value") is not None:
        return [observation]
    if isinstance(observation, (list, tuple)) and len(observation) > 0 and all(
        isinstance(o, (dict, pd.DataFrame)) for o in observation
    ):
        return list(observation)
    return [observation]
