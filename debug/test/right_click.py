import pyvista as pv
import vtk

# 1. 初始化 Plotter
pl = pv.Plotter()
mesh = pv.Cube()
pl.add_mesh(mesh, color="lightblue", show_edges=True)

# 2. 手动创建一个 Picker 实例 (用于将屏幕点击转换为 3D 坐标)
prop_picker = vtk.vtkPropPicker()

def on_right_click(obj, event):
    # 获取鼠标点击的屏幕像素位置 (x, y)
    click_pos = pl.iren.get_event_position()
    
    # 使用 picker 进行拾取
    # 参数：(x, y, z, renderer)
    prop_picker.Pick(click_pos[0], click_pos[1], 0, pl.renderer)
    
    # 获取拾取到的世界坐标
    world_pos = prop_picker.GetPickPosition()
    
    # 逻辑判断：如果点击到了物体 (PickPosition 不为 0,0,0)
    if any(world_pos):
        print(f"右键点击成功! 坐标: {world_pos}")
        
        # 差异化功能示例：在点击位置放置一个标签或小球
        marker = pv.Sphere(radius=0.05, center=world_pos)
        pl.add_mesh(marker, color="orange", name=f"pt_{world_pos}")
    else:
        print("未点击到任何物体")

# 3. 绑定右键事件到交互器 (iren)
pl.iren.add_observer("RightButtonPressEvent", on_right_click)

pl.show()