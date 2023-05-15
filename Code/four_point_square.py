import os
import numpy as np


from_path = './dummy'
to_path = './dummy'


dt = np.loadtxt(os.path.join(from_path, 'X_data_four_point.txt'), delimiter=" ", dtype=float)
new_dt = []

for i in range(len(dt)):
    new_dt.append([ele*ele for ele in dt[i]])
        
new_dt = np.array(new_dt)
np.savetxt(f'./{to_path}/X_data_four_point_square.txt', new_dt, fmt='%.2f', delimiter=' ')