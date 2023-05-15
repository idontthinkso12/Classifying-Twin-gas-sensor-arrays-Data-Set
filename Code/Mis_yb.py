import os
import numpy as np
from tqdm import tqdm
from collections import Counter

class Ori_t_generator:
    def __init__(self) -> None:
        pass

    def generate(self, from_dir, to_dir):
        
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)
        
        filenames = os.listdir(from_dir)
        
        if len(filenames) == 0:
            for i in tqdm(range(len(filenames))):
            
                dt = np.loadtxt(os.path.join(from_dir, filenames[i]), delimiter="\t", dtype=float)

                if dt.shape[0] >= 50000:
                    dt = dt[:50000, 1:]
                    np.savetxt(f'{to_dir}/{filenames[i]}', dt, fmt='%.2f', delimiter='\t')
        else:
            print('The target directory is non-empty.')

    def check(self, to_dir):
        filenames = os.listdir(to_dir)
        ct = Counter()
        for i in tqdm(range(len(filenames))):
            dt = np.loadtxt(os.path.join(to_dir, filenames[i]), delimiter="\t", dtype=float)
            ct[dt.shape[0]] += 1
        print(ct)
    
    def halve(self, from_dir, to_dir):
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)
        
        filenames = os.listdir(from_dir)
        for i in tqdm(range(len(filenames))):
            dt = np.loadtxt(os.path.join(from_dir, filenames[i]), delimiter="\t", dtype=float)
            dt = dt[:25000, :]
            np.savetxt(f'{to_dir}/{filenames[i]}', dt, fmt='%.2f', delimiter='\t')

if __name__ == '__main__':
#    g1 = Ori_t_generator()
#    from_dir = '../data1'
#    to_dir = './ori_t'
#    g1.generate(from_dir, to_dir)
#    g1.check(to_dir)

    g2 = Ori_t_generator()
    from_dir = './ori_t'
    to_dir = './ori_t_half'
    g2.halve(from_dir, to_dir)
    g2.check(to_dir)
