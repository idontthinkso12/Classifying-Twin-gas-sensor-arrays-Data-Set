import os
import numpy as np


from_path = './dummy'
to_path = './dummy'


dt = np.loadtxt(os.path.join(from_path, 'X_data_50_Half.txt'), delimiter=" ", dtype=float)
new_dt = []

for i in range(len(dt)):
    s5 = []
    s7 = []
    for j in range(0, len(dt[i]), 8):
        s5.append(dt[i][j+4])
        s7.append(dt[i][j+6])    
    new_dt.append([min(s5[:25]), max(s5[:25]),min(s5[25:]), max(s5[25:]),min(s7[:25]), max(s7[:25]),min(s7[25:]), max(s7[25:])])


new_dt = np.array(new_dt)
np.savetxt(f'./{to_path}/X_data_four_point.txt', new_dt, fmt='%.2f', delimiter=' ')