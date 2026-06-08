# Headless (Server) Installation — Known Issues

## 背景

为了在 HPCC 等无显示器的服务器上使用 nematics3d，需要 osmesa 版本的 VTK（基于 Mesa 软件渲染，不依赖 X server）。
2026-06-08 尝试通过 pyproject.toml extras 实现 `pip install ".[headless]"` 的一键安装，发现两个根本性问题。

---

## 问题一：`vtk-osmesa` 不在 PyPI 上

`vtk-osmesa` 这个包名在 PyPI 上不存在。osmesa 版本的 VTK 只能通过 **conda-forge** 安装：

```bash
conda install -c conda-forge "vtk-base=9.3.1=osmesa*"
```

**后果**：pyproject.toml 里的 `[headless]` extra 目前是空的（已移除 `vtk-osmesa` 那行），
纯 `pip install ".[headless]"` 无法完成 VTK 的安装。

**待办**：
- 调研是否有可行的 pip-only 方案（例如通过 `--extra-index-url` 指向某个托管 osmesa wheel 的私有源）
- 或者接受现状，在文档里明确说明必须先用 conda 装 VTK，再用 pip 装其余部分

---

## 问题二：`pyvistaqt` 是顶层 import，headless 也必须装

`src/nematics3d/classes/q_field_object.py` 第 103 行：

```python
from pyvistaqt import BackgroundPlotter
```

这是模块级别的 import，不是懒加载。导致 headless 环境即使不使用交互功能，也必须安装 `pyvistaqt` 和 `PyQt5`，否则 `import nematics3d` 直接失败。

**后果**：`gui` 和 `headless` 在代码层面没有真正分离。headless 环境目前还是要装 PyQt5 全家桶。

**待办**：
- 把 `pyvistaqt` 相关 import 改成条件导入，只在用到交互功能时才 import：
  ```python
  # 在函数/方法内部，而非模块顶层
  def show_interactive(...):
      from pyvistaqt import BackgroundPlotter
      ...
  ```
- 或者用 `TYPE_CHECKING` + `Optional` 标注，把 import 推迟到运行时
- 改完之后 headless 环境可以不装 PyQt5，真正做到与 gui 分离

---

## 当前 HPCC 上的实际安装方式（`Nematics3D_headless` 环境）

```bash
conda create -n Nematics3D_headless python=3.12
conda activate Nematics3D_headless
conda install -c conda-forge "vtk-base=9.3.1=osmesa*"
pip install pyvista==0.46.4 pyvistaqt==0.11.3 PyQt5==5.15.11 qtpy==2.4.1 scipy==1.16.0
pip install --no-deps /work/yingyouma/Nematics3D
```

渲染测试通过（`vtkOSOpenGLRenderWindow`，Mesa llvmpipe 软件渲染）。
