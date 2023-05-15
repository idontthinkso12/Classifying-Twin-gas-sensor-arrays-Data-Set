import os
import numpy as np
from scipy.signal import convolve2d
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class Dataset_prep:
    def __init__(self, to_m = 50, folder_path_ori = './ori_t') -> None:
        self.FullOrHalf = 'Half'
        self.modes = {0:'Random_Sampling', 1: 'Non_Probability_Sampling', 2: 'Mean', 3: 'Gaussian', 4: 'Square'}
        self.folder_path_ori = folder_path_ori
        self.to_m = to_m
        self.k = 631
        self.m_ori = 25000
        compress_factor, resi = divmod(self.m_ori, self.to_m)
        self.c = compress_factor
        self.from_m = self.m_ori - resi
        self.test_size_proportions = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
       
    def prepare_datasets(self):
        new_folders = [ f'./data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[0]}',
                        f'./data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[1]}',
                        f'./data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[2]}',
                        f'./data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[3]}',
                        f'./data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[4]}']
        
        for new_folder in new_folders:
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
        
        to_paths = [    f'./training_data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[0]}',
                        f'./training_data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[1]}',
                        f'./training_data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[2]}',
                        f'./training_data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[3]}',
                        f'./training_data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[4]}']

        for to_path in to_paths:
            if not os.path.exists(to_path):
                os.makedirs(to_path)

        filenames = os.listdir(self.folder_path_ori)

        flatten_res0 = []
        flatten_res1 = []
        flatten_res2 = []
        flatten_res3 = []
        flatten_res4 = []
        y_res = []
        # construct new data
        for i in range(len(filenames)):
            if os.path.exists(f'{new_folders[4]}/{filenames[i]}'):
                continue 

            if 'GCO' in filenames[i]:
                    y_res.append(1)
            elif 'GEa' in filenames[i]:
                y_res.append(2)
            elif 'GEy' in filenames[i]:
                y_res.append(3)
            else:
                y_res.append(4)

            # read data
            dt = np.loadtxt(os.path.join(self.folder_path_ori, filenames[i]), delimiter="\t", dtype=float)
            
            compress_k_0 = np.zeros(self.c).reshape((self.c, 1))
            compress_k_0[np.random.randint(self.c)] = 1
            res0 = convolve2d(dt, compress_k_0, mode='valid')[::self.c, ::1]
            np.savetxt(f'{new_folders[0]}/{filenames[i]}', res0, fmt='%.2f', delimiter='\t')
            flatten_res0.append(res0.flatten())

            compress_k_1 = np.zeros(self.c).reshape((self.c, 1))
            compress_k_1[-1] = 1
            res1 = convolve2d(dt, compress_k_1, mode='valid')[::self.c, ::1]
            np.savetxt(f'{new_folders[1]}/{filenames[i]}', res1, fmt='%.2f', delimiter='\t')
            flatten_res1.append(res1.flatten())


            compress_k_2 = np.ones(self.c).reshape((self.c, 1))
            res2 = convolve2d(dt, compress_k_2, mode='valid')[::self.c, ::1]
            res2 = res2/self.c
            np.savetxt(f'{new_folders[2]}/{filenames[i]}', res2, fmt='%.2f', delimiter='\t')
            flatten_res2.append(res2.flatten())


            compress_k_3 = norm.pdf(np.arange(-4, 4, 8/self.c), 0, 1).reshape((self.c, 1))
            res3 = convolve2d(dt, compress_k_3, mode='valid')[::self.c, ::1]
            res3 = res3/self.c
            np.savetxt(f'{new_folders[3]}/{filenames[i]}', res3, fmt='%.2f', delimiter='\t')
            flatten_res3.append(res3.flatten())


            compress_k_4 = np.ones(self.c).reshape((self.c, 1))
            res4 = convolve2d(np.square(dt), compress_k_4, mode='valid')[::self.c, ::1]
            np.savetxt(f'{new_folders[4]}/{filenames[i]}', res4, fmt='%.2f', delimiter='\t')
            flatten_res4.append(res4.flatten())    
        
        y_res = np.array(y_res)
        np.savetxt(f'./training_data/y_data_{self.to_m}_{self.FullOrHalf}.txt', y_res, fmt='%d', delimiter=' ')

        flatten_res0 = np.array(flatten_res0)
        np.savetxt(f'{to_paths[0]}/X_data.txt', flatten_res0, fmt='%.2f', delimiter=' ')
        self.Quick_test(flatten_res0, y_res, 0)
        
        flatten_res1 = np.array(flatten_res1)
        np.savetxt(f'{to_paths[1]}/X_data.txt', flatten_res1, fmt='%.2f', delimiter=' ')
        self.Quick_test(flatten_res1, y_res, 1)

        flatten_res2 = np.array(flatten_res2)
        np.savetxt(f'{to_paths[2]}/X_data.txt', flatten_res2, fmt='%.2f', delimiter=' ')
        self.Quick_test(flatten_res2, y_res, 2)

        flatten_res3 = np.array(flatten_res3)
        np.savetxt(f'{to_paths[3]}/X_data.txt', flatten_res3, fmt='%.2f', delimiter=' ')
        self.Quick_test(flatten_res3, y_res, 3)

        flatten_res4 = np.array(flatten_res4)
        np.savetxt(f'{to_paths[4]}/X_data.txt', flatten_res4, fmt='%.2f', delimiter=' ')
        self.Quick_test(flatten_res4, y_res, 4)

    def Flatten_data(self):
        new_folders = [ f'./data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[0]}',
                        f'./data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[1]}',
                        f'./data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[2]}',
                        f'./data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[3]}',
                        f'./data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[4]}']
                
        to_paths = [    f'./training_data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[0]}',
                        f'./training_data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[1]}',
                        f'./training_data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[2]}',
                        f'./training_data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[3]}',
                        f'./training_data/compressed_data_{self.to_m}_{self.FullOrHalf}/{self.modes[4]}']

        for i in range(5):
            folder_path = new_folders[i]
            files_yb = os.listdir(folder_path)
            res = []
            for ii in range(len(files_yb)):
                dt = np.loadtxt(os.path.join(folder_path, files_yb[ii]), delimiter="\t", dtype=float)
                res.append(dt.flatten())
            res = np.array(res)
            to_path = to_paths[i]
            np.savetxt(f'{to_path}/X_data.txt', res,fmt='%.2f', delimiter=' ')


    def Quick_test(self, X, y, compress_mode):
        result_path = f'./Test_results/{self.FullOrHalf}'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        acc_res = {} 
        for i in range(6):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size_proportions[i])
            acc_res[self.test_size_proportions[i]] = []
            for k_val in range(5,0,-1):
                cls1 = KNeighborsClassifier(k_val)
                cls1.fit(X_train, y_train)
                score = cls1.score(X_test, y_test)
                acc_res[self.test_size_proportions[i]].append(round(score,4))
        with open(f'{result_path}/{self.to_m}_{self.modes[compress_mode]}.txt','w') as f:
            f.write(str(acc_res))
        f.close()

    def Test_with_load(self, X_file_path, y_file_path, compress_mode):
        X = np.loadtxt(X_file_path, delimiter=" ", dtype=float)
        y = np.loadtxt(y_file_path, delimiter=" ", dtype=float)
        print(X.shape)
        print(y.shape)
        

        result_path = f'./Test_results/{self.FullOrHalf}'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        acc_res = {} 
        for i in range(6):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size_proportions[i])
            acc_res[self.test_size_proportions[i]] = []
            for k_val in range(5,0,-1):
                cls1 = KNeighborsClassifier(k_val)
                cls1.fit(X_train, y_train)
                score = cls1.score(X_test, y_test)
                acc_res[self.test_size_proportions[i]].append(round(score,4))
        with open(f'{result_path}/m_{self.to_m}_{self.modes[compress_mode]}.txt','w') as f:
            f.write(str(acc_res))
        f.close()