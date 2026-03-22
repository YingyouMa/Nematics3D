from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(r'D:\Document\GitHub\Nematics3D\src')))
import Nematics3D

DATA_DIR = Path(r'D:\Document\GitHub\Nematics3D\example\data')
n = np.load(DATA_DIR / 'n_example_global.npy')
S = np.load(DATA_DIR / 'S_example_global.npy')
Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=True, name='testQ')
Q.act_lines_smooth()
line = next(obj for obj in Q.objects if getattr(obj, 'name', None) == 'disclination line 5')
smooth = line.smooths[0]
raw = line._raw_defect_indices
calc = smooth._calc_coords
res = smooth._calc_result
print('raw len:', len(raw))
print('raw first==last:', np.allclose(raw[0], raw[-1]))
print('raw first:', raw[0])
print('raw last :', raw[-1])
print('calc len:', len(calc))
print('calc first==last:', np.allclose(calc[0], calc[-1]))
print('calc first:', calc[0])
print('calc last :', calc[-1])
print('result len:', len(res))
print('result first==last:', np.allclose(res[0], res[-1]))
print('result first:', res[0])
print('result last :', res[-1])
