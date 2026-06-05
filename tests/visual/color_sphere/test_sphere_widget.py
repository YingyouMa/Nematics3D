import sys
sys.path.insert(0, 'D:/Document/GitHub/Nematics3D/src')
from nematics3d.classes.visual.plot_figure import PlotFigure

# Verify menu entries are added without error (off-screen skips menu)
fig = PlotFigure(is_off_screen=True, size=(800, 800))

# Manually call the menu helper to check it doesn't crash when main_menu is None
fig._helper_setup_orientation_menu()

# Check sphere_axes widget works
fig.act_add_axes_widget(style='sphere_axes', viewport=(0.0, 0.0, 0.35, 0.35))
fig.pl.render()
fig.act_savefig(
    'D:/Document/GitHub/Nematics3D/tests/visual/color_sphere/sphere_widget_test.png',
    scale=1, window_size=(800, 800),
)
fig.act_close()
print('OK')
