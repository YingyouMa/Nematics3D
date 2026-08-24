"""Build configuration for the optional compiled Q-diagonalization backend."""

import os
import sys

import numpy as np
from setuptools import Extension, setup


def _qdiag_compile_args():
    if sys.platform == "win32":
        return ["/O2"]
    return ["-O3", "-std=c11"]


def _qdiag_libraries():
    if sys.platform.startswith("linux"):
        return ["m"]
    return []


setup(
    ext_modules=[
        Extension(
            "nematics3d.analysis.q_diagonalization._core",
            sources=[
                os.path.join(
                    "src",
                    "nematics3d",
                    "analysis",
                    "q_diagonalization",
                    "qdiag_module.c",
                ),
                os.path.join(
                    "src",
                    "nematics3d",
                    "analysis",
                    "q_diagonalization",
                    "qdiag_kernel.c",
                ),
            ],
            depends=[
                os.path.join(
                    "src",
                    "nematics3d",
                    "analysis",
                    "q_diagonalization",
                    "qdiag_kernel.h",
                )
            ],
            include_dirs=[
                np.get_include(),
                os.path.join("src", "nematics3d", "analysis", "q_diagonalization"),
            ],
            libraries=_qdiag_libraries(),
            extra_compile_args=_qdiag_compile_args(),
            language="c",
            optional=True,
        )
    ]
)
