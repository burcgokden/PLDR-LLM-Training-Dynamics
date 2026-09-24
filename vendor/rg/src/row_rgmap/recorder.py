"""Native row-energy capture for instrumented PLDR training loops."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def _as_matrix(value, map_id: str) -> np.ndarray:
    """Detach a NumPy-like or PyTorch-like value and return float64 on CPU."""

    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
        is_complex = getattr(candidate, "is_complex", None)
        if callable(is_complex) and is_complex():
            raise ValueError(f"row map {map_id!r} must be real")
        if hasattr(candidate, "double"):
            candidate = candidate.double()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    raw = np.asarray(candidate)
    if np.iscomplexobj(raw):
        raise ValueError(f"row map {map_id!r} must be real")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"row map {map_id!r} must be a nonempty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"row map {map_id!r} must be finite")
    return matrix


def _matrix_energy(value: np.ndarray) -> float:
    if np.all(value == value[0:1, :]):
        return 0.0
    centered = value - np.mean(value, axis=0, keepdims=True, dtype=np.float64)
    energy = float(np.sum(centered * centered, dtype=np.float64))
    if not np.isfinite(energy) or energy < 0.0:
        raise ValueError("row-centered energy is not finite and nonnegative")
    return energy


def row_centered_energy(matrix) -> float:
    """Return the squared Frobenius norm after row centering in float64."""

    return _matrix_energy(_as_matrix(matrix, "<direct>"))


class RowEnergyRecorder:
    """Capture a fixed row-map registry at consecutive optimizer updates.

    Values may be NumPy arrays or detached-compatible tensor objects.  A
    training producer should call record after each optimizer update,
    including the initial state if the intended first edge starts there.
    """

    def __init__(self, map_ids: Sequence[str]) -> None:
        resolved = tuple(str(value) for value in map_ids)
        if not resolved or any(not value for value in resolved):
            raise ValueError("map_ids must be nonempty strings")
        if len(set(resolved)) != len(resolved):
            raise ValueError("map_ids must be unique")
        self.map_ids = resolved
        self._steps: list[int] = []
        self._energies: list[list[float]] = []
        self._shapes: tuple[tuple[int, int], ...] | None = None

    def record(self, step: int, row_maps: Mapping[str, object]) -> None:
        if not isinstance(step, (int, np.integer)) or isinstance(step, bool):
            raise ValueError("step must be an integer")
        step = int(step)
        if self._steps and step != self._steps[-1] + 1:
            raise ValueError("recorder steps must be consecutive")
        missing = [name for name in self.map_ids if name not in row_maps]
        if missing:
            raise ValueError(f"row-map registry entries are missing: {missing}")
        matrices = tuple(_as_matrix(row_maps[name], name) for name in self.map_ids)
        shapes = tuple(matrix.shape for matrix in matrices)
        if self._shapes is None:
            self._shapes = shapes
        elif shapes != self._shapes:
            raise ValueError("row-map shapes changed within a trajectory")
        energies = [_matrix_energy(matrix) for matrix in matrices]
        if not np.isfinite(energies).all() or any(
            energy < 0.0 for energy in energies
        ):
            raise ValueError("captured energies must be finite and nonnegative")
        self._steps.append(step)
        self._energies.append(energies)

    def state_dict(self) -> dict[str, object]:
        """Return a compact checkpoint state for interruption-safe producers."""

        return {
            "map_ids": list(self.map_ids),
            "steps": np.asarray(self._steps, dtype=np.int64),
            "energies": np.asarray(self._energies, dtype=np.float64),
            "shapes": list(self._shapes) if self._shapes is not None else None,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a state emitted by :meth:`state_dict` into an empty recorder."""

        if self._steps or self._energies or self._shapes is not None:
            raise ValueError("recorder restore requires a fresh recorder")
        if tuple(str(value) for value in state.get("map_ids", [])) != self.map_ids:
            raise ValueError("checkpoint map registry differs from recorder")
        steps = np.asarray(state.get("steps"))
        energies = np.asarray(state.get("energies"))
        if steps.ndim != 1 or not np.issubdtype(steps.dtype, np.integer):
            raise ValueError("checkpoint steps are invalid")
        if energies.shape != (len(steps), len(self.map_ids)):
            raise ValueError("checkpoint energies have the wrong shape")
        if energies.dtype != np.dtype(np.float64):
            raise ValueError("checkpoint energies must be float64")
        if not np.isfinite(energies).all() or np.any(energies < 0.0):
            raise ValueError("checkpoint energies must be finite and nonnegative")
        step_values = [int(value) for value in steps.tolist()]
        if any(right != left + 1 for left, right in zip(step_values, step_values[1:])):
            raise ValueError("checkpoint steps must be consecutive")
        shapes = state.get("shapes")
        if shapes is None:
            if len(steps):
                raise ValueError("nonempty checkpoint lacks row-map shapes")
            restored_shapes = None
        else:
            restored_shapes = tuple(
                tuple(int(item) for item in shape) for shape in shapes
            )
            if len(restored_shapes) != len(self.map_ids) or any(
                len(shape) != 2 or min(shape) < 1 for shape in restored_shapes
            ):
                raise ValueError("checkpoint row-map shapes are invalid")
        self._steps = step_values
        self._energies = energies.tolist()
        self._shapes = restored_shapes

    def write_npz(self, path: str | Path, trajectory_id: str) -> Path:
        if len(self._steps) < 2:
            raise ValueError("at least two recorded states are required")
        trajectory_id = str(trajectory_id)
        if not trajectory_id:
            raise ValueError("trajectory_id must be nonempty")
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("energy artifacts must use the .npz suffix")
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        energies = np.asarray(self._energies, dtype=np.float64)[None, :, :]
        np.savez_compressed(
            destination,
            energies=energies,
            steps=np.asarray(self._steps, dtype=np.int64),
            trajectory_ids=np.asarray([trajectory_id]),
            map_ids=np.asarray(self.map_ids),
        )
        return destination
