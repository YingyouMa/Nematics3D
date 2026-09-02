import sys
from pathlib import Path
import types
import unittest
from dataclasses import dataclass

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.core.registry_base import RegistryBase
from nematics3d.core.result_base import ResultBase
from nematics3d.classes.disclination_line import DisclinationLine, InputLine
from nematics3d.classes.smoothed_line import SmoothedLine, SmoothedLineFunc


@dataclass(repr=False)
class ScalarResult(ResultBase):
    value: float


def scalar_result(u):
    return ScalarResult(float(u))


def doubled_scalar_result(u):
    return ScalarResult(2.0 * float(u))


class WrapLineFuncModeLine(SmoothedLine):
    @property
    def linefunc_mode(self):
        return "wrap"


def build_line():
    x = np.linspace(0.0, 1.0, 60)
    return np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))


class TestSmoothedLineFuncRegistry(unittest.TestCase):
    def test_linefunc_registry_is_created_with_owner(self):
        line = SmoothedLine(build_line(), window_length=5, min_line_length=2)

        self.assertIsInstance(line.linefuncs, RegistryBase)
        self.assertIs(line.linefuncs, line.entity_linefuncs)
        self.assertIs(line.linefuncs.owner, line)
        self.assertEqual(len(line.linefuncs), 0)

    def test_act_create_linefunc_registers_default_names(self):
        line = SmoothedLine(build_line(), window_length=5, min_line_length=2)

        first = line.act_create_linefunc(scalar_result, [0, 50, 100])
        second = line.act_create_linefunc(doubled_scalar_result, [0, 50, 100])

        self.assertIsInstance(first, SmoothedLineFunc)
        self.assertEqual(first.name, "line_func_0")
        self.assertEqual(second.name, "line_func_1")
        self.assertIs(first.owner, line)
        self.assertIs(first.registry, line.linefuncs)
        self.assertIs(line.linefuncs[0], first)
        self.assertIs(line.linefuncs["line_func_1"], second)
        self.assertEqual(line.impl_linefunc_count, 2)

    def test_explicit_name_still_advances_default_name_counter(self):
        line = SmoothedLine(build_line(), window_length=5, min_line_length=2)

        named = line.act_create_linefunc(scalar_result, [0, 50, 100], name="density")
        default = line.act_create_linefunc(scalar_result, [0, 50, 100])

        self.assertEqual(named.name, "density")
        self.assertEqual(default.name, "line_func_1")
        self.assertEqual(line.impl_linefunc_count, 2)

    def test_linefunc_mode_can_differ_from_smoothing_mode(self):
        line = WrapLineFuncModeLine(
            build_line(),
            window_length=5,
            min_line_length=2,
            mode="interp",
        )

        linefunc = line.act_create_linefunc(
            scalar_result,
            np.arange(0, 100, 25),
            name="wrapped_scalar",
        )

        self.assertEqual(line.opts.mode, "interp")
        self.assertEqual(line.linefunc_mode, "wrap")
        self.assertEqual(linefunc.impl_owner_opts_snapshot["mode"], "interp")
        self.assertEqual(linefunc.impl_owner_opts_snapshot["linefunc_mode"], "wrap")
        self.assertIn("mode='wrap'", repr(linefunc))
        self.assertAlmostEqual(float(linefunc(105)), float(linefunc(5)))

    def test_cross_disclination_linefunc_mode_is_periodic(self):
        indices = np.column_stack(
            (
                np.arange(8, 19),
                np.full(11, 0.5),
                np.full(11, 0.5),
            )
        )
        line = DisclinationLine(
            InputLine(
                defect_indices=indices,
                box_size_periodic_index=(10, 10, 10),
            ),
            is_sorted=True,
        )

        smooth = line.act_smooth(window_length=5, min_line_length=2)
        linefunc = smooth.act_create_linefunc(
            scalar_result,
            np.arange(0, 100, 25),
            name="cross_scalar",
        )

        self.assertEqual(line.kind, "cross")
        self.assertEqual(smooth.opts.mode, "interp")
        self.assertEqual(smooth.linefunc_mode, "wrap")
        self.assertEqual(linefunc.impl_owner_opts_snapshot["linefunc_mode"], "wrap")


if __name__ == "__main__":
    unittest.main()
