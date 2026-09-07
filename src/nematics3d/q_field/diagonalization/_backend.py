"""Optional compiled backend for Q-tensor diagonalization."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from . import _core
except ImportError as error:
    _core = None
    _QDIAG_IMPORT_ERROR = error
else:
    _QDIAG_IMPORT_ERROR = None


_AUTO_MAX_WORKERS = 8
_AUTO_TENSORS_PER_WORKER = 100_000


def is_c_backend_available():
    """Return whether the compiled Q-diagonalization extension is importable."""
    return _core is not None


def c_backend_import_error():
    """Return the import error that made the compiled backend unavailable."""
    return _QDIAG_IMPORT_ERROR


def _resolve_worker_count(worker_count, tensor_count):
    if worker_count is None:
        useful_workers = max(
            1,
            (tensor_count + _AUTO_TENSORS_PER_WORKER - 1) // _AUTO_TENSORS_PER_WORKER,
        )
        return min(os.cpu_count() or 1, _AUTO_MAX_WORKERS, useful_workers)
    return min(worker_count, max(1, tensor_count))


def diagonalize_qfield5(qfield5, *, is_biaxial, worker_count):
    """Diagonalize a validated QField5 through the compiled backend."""
    if _core is None:
        raise ImportError(
            "The compiled Nematics3D Q-diagonalization backend is unavailable."
        ) from _QDIAG_IMPORT_ERROR

    if qfield5.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        qfield5 = np.asarray(qfield5, dtype=np.float64)
    qfield5 = np.ascontiguousarray(qfield5)
    shape = qfield5.shape[:-1]
    flat_q = qfield5.reshape(-1, 5)
    tensor_count = flat_q.shape[0]
    actual_worker_count = _resolve_worker_count(worker_count, tensor_count)

    if is_biaxial:
        eigenvalues = np.empty((tensor_count, 3), dtype=np.float64)
        eigenvectors = np.empty((tensor_count, 3, 3), dtype=np.float64)
        solve_into = _core.eigh_qfield5_into
    else:
        eigenvalues = np.empty(tensor_count, dtype=np.float64)
        eigenvectors = np.empty((tensor_count, 3), dtype=np.float64)
        solve_into = _core.dominant_qfield5_into

    if actual_worker_count == 1:
        solve_into(flat_q, eigenvalues, eigenvectors)
    else:
        boundaries = [
            tensor_count * index // actual_worker_count
            for index in range(actual_worker_count + 1)
        ]
        with ThreadPoolExecutor(max_workers=actual_worker_count) as executor:
            futures = [
                executor.submit(
                    solve_into,
                    flat_q[start:stop],
                    eigenvalues[start:stop],
                    eigenvectors[start:stop],
                )
                for start, stop in zip(boundaries[:-1], boundaries[1:])
            ]
            for future in futures:
                future.result()

    if is_biaxial:
        return (
            eigenvalues.reshape(shape + (3,)),
            eigenvectors.reshape(shape + (3, 3)),
            actual_worker_count,
        )
    return (
        eigenvalues.reshape(shape),
        eigenvectors.reshape(shape + (3,)),
        actual_worker_count,
    )
