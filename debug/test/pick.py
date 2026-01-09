import pyvista as pv
import numpy as np

# --- scene ---
pl = pv.Plotter()
pl.add_axes()

points = np.array([[0, 0, 0], [1, 2, 0], [2, 0, 0]])
line = pv.MultipleLines(points=points)
line_actor = pl.add_mesh(line, color="black", line_width=10, name="my_line")

orig_color = line_actor.prop.color  # 记录原色（RGB tuple）
sphere_name = "pick_sphere"
picked = {"active": False}          # 简单状态位，避免重复操作


def reset_selection():
    """恢复线条颜色并移除小球。"""
    line_actor.prop.color = orig_color
    pl.remove_actor(sphere_name, reset_camera=False)
    picked["active"] = False
    pl.render()


def on_pick(point, picker):
    """
    point: (x, y, z) picked point (可能是点到mesh，也可能是点到3D窗口)
    picker: VTK picker（用来判断点到的是哪个actor）
    """
    actor = picker.GetActor() if picker is not None else None

    # 点到线：高亮 + 小球 + 打印坐标
    if actor is line_actor:
        line_actor.prop.color = "red"
        pl.remove_actor(sphere_name, reset_camera=False)  # 先清一次，确保只剩一个球
        sph = pv.Sphere(radius=0.03, center=point)
        pl.add_mesh(sph, color="yellow", name=sphere_name)
        print(f"Picked point: {tuple(float(x) for x in point)}")
        picked["active"] = True
        pl.render()
        return

    # 点到别处：如果之前选中过，就清除
    if picked["active"]:
        reset_selection()


# --- picking ---
pl.enable_point_picking(
    callback=on_pick,
    left_clicking=True,       # 左键点选
    pickable_window=True,     # 允许点到空白区域（用于“点到别处就复位”）
    use_picker=True,          # 回调额外传入 picker
    show_point=False,         # 不用内置的粉色点
    picker="cell",            # 线这种细几何更建议用 cell picker + tolerance
    tolerance=0.03,
    show_message=False,
)

pl.show(interactive_update=True)

# import pyvista as pv
# import numpy as np

# # 1. 创建线段
# points = np.array([[0, 0, 0], [1, 2, 0], [2, 0, 0]])
# line = pv.MultipleLines(points=points)

# pl = pv.Plotter()

# # 2. 添加模型，给它一个唯一的名称 (name) 方便后续寻找
# line_actor = pl.add_mesh(line, color="cyan", line_width=5, name="my_line")

# # 3. 定义回调函数
# def pick_callback(actor):
#     # actor 就是被选中的对象
#     if actor == line_actor:
#         print("线段被选中了！")
#         # 改变选中后的样式
#         line_actor.prop.color = "red"      # 变红
#         line_actor.prop.line_width = 10    # 变粗
#     else:
#         # 如果点到了别处（或没点中），恢复原样
#         line_actor.prop.color = "cyan"
#         line_actor.prop.line_width = 5

# # 4. 启用网格拾取
# # left_clicking=True 表示直接用鼠标左键点击，不需要按 'P' 键
# pl.enable_mesh_picking(callback=pick_callback, left_clicking=True, show_message=True)

# pl.view_xy()
# pl.show(interactive_update=True)