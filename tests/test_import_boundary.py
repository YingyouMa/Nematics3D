import subprocess
import sys


def test_root_import_does_not_load_qt_stack():
    code = (
        "import sys; import nematics3d; "
        "assert 'qtpy' not in sys.modules, sorted(k for k in sys.modules if k.startswith('qt')); "
        "assert 'pyvistaqt' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_qfieldobject_import_does_not_load_qt_stack():
    code = (
        "import sys; from nematics3d import QFieldObject; "
        "assert 'qtpy' not in sys.modules, sorted(k for k in sys.modules if k.startswith('qt')); "
        "assert 'pyvistaqt' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_plot_figure_module_import_does_not_load_qt_stack():
    code = (
        "import sys; import nematics3d.visual.plot_figure; "
        "assert 'qtpy' not in sys.modules, sorted(k for k in sys.modules if k.startswith('qt')); "
        "assert 'pyvistaqt' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
