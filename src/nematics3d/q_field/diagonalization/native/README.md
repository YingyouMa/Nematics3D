# Native Q-diagonalization backend

This directory contains the compiled implementation sources for
`nematics3d.q_field.diagonalization._core`.

Platform-specific extension binaries such as `_core.cp312-win_amd64.pyd` or
`_core.cpython-312-x86_64-linux-gnu.so` are build artifacts. They are not
source files and should not be committed here. The build installs the extension
beside the Python diagonalization package so `_backend.py` can import it with
`from . import _core`.
