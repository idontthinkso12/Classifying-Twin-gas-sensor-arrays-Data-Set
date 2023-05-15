import os
import numpy as np
from tqdm import tqdm
from scipy.signal import convolve2d
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class Dataset_prep:
    def __init__(self, to_m = 50, folder_path_ori = './ori_t') -> None:
        self.modes = {0:'Random_Sampling', 1: 'Non_Probability_Sampling', 2: 'Mean', 3: 'Gaussian', 4: 'Square'}
        self.folder_path_ori = folder_path_ori
        self.to_m = to_m

        if folder_path_ori == './ori_t':
            self.k = 631
            self.m_ori = 50000
        else:
            # all samples have the same size
            filenames = os.listdir(folder_path_ori)
            self.k = len(filenames) 
            dt = np.loadtxt(os.path.join(self.folder_path_ori, filenames[0]), delimiter="\t", dtype=float)
            self.m_ori = dt.shape[0]
        compress_factor, resi = divmod(self.m_ori, self.to_m)
        self.c = compress_factor
        self.from_m = self.m_ori - resi
        #print(f'm: {self.m_ori} ----> {self.from_m} ----> {self.to_m}\nn: 8\nk: {self.k}\nc: {self.c}')        

    def compress_dataset(self, compress_modes = [0,1,2,3,4]):
        self.compress_modes = compress_modes
        for compress_mode in compress_modes:
            new_folder = f'./data/compressed_data_{self.to_m}/{self.modes[compress_mode]}'
            if not os.path.exists(new_folder):
                print(f'New folder created! {new_folder}')
                os.makedirs(new_folder)
            filenames = os.listdir(self.folder_path_ori)

        # construct the compress_kernel
            if compress_mode == 0:
                compress_k = np.zeros(self.c).reshape((self.c, 1))
                compress_k[np.random.randint(self.c)] = 1
            
            elif compress_mode == 1:
                compress_k = np.zeros(self.c).reshape((self.c, 1))
                #compress_k[0] = 1
                compress_k[-1] = 1
            
            elif compress_mode == 2 or compress_mode == 4:
                compress_k = np.ones(self.c).reshape((self.c, 1))
            elif compress_mode == 3: # Gaussian kernel
                compress_k = norm.pdf(np.arange(-4, 4, 8/self.c), 0, 1).reshape((self.c, 1))

            # construct new data
            for i in tqdm(range(len(filenames))):
                # read data
                dt = np.loadtxt(os.path.join(self.folder_path_ori, filenames[i]), delimiter="\t", dtype=float)
                if compress_mode in (0, 1):
                    res = convolve2d(dt, compress_k, mode='valid')[::self.c, ::1]
                elif compress_mode in (2, 3):
                    res = convolve2d(dt, compress_k, mode='valid')[::self.c, ::1]
                    res = res/self.c
                elif compress_mode == 4:
                    dt2 = np.square(dt) # sum of square
                    res = convolve2d(dt2, compress_k, mode='valid')[::self.c, ::1]

                # store compressed data
                np.savetxt(f'{new_folder}/{filenames[i]}', res, fmt='%.2f', delimiter='\t')
            
    def construct_trainX(self):
        for compress_mode in self.compress_modes:
            folder_path = f'./data/compressed_data_{self.to_m}/{self.modes[compress_mode]}'
            files_yb = os.listdir(folder_path)
            res = []
            for i in tqdm(range(len(files_yb))):
                dt = np.loadtxt(os.path.join(folder_path, files_yb[i]), delimiter="\t", dtype=float)
                res.append(dt.flatten())
            res = np.array(res)
            to_path = f'./training_data/compressed_data_{self.to_m}/{self.modes[compress_mode]}'
            if not os.path.exists(to_path):
                os.makedirs(to_path)
            np.savetxt(f'{to_path}/X_data.txt', res,fmt='%.2f', delimiter=' ')
            #return res

    def construct_trainy(self):
        to_path = './training_data'
        if os.path.exists(f'{to_path}/y_data.txt'):
            return
        else:
            folder_path = f'./data/compressed_data_{self.to_m}/{self.modes[self.compress_modes[0]]}'
            files_yb = os.listdir(folder_path)
            res = []
            for i in tqdm(range(len(files_yb))):
                if 'GCO' in files_yb[i]:
                    res.append(1)
                elif 'GEa' in files_yb[i]:
                    res.append(2)
                elif 'GEy' in files_yb[i]:
                    res.append(3)
                else:
                    res.append(4)
            res = np.array(res)
            #print(res.shape)
            if not os.path.exists(to_path):
                os.makedirs(to_path)
            np.savetxt(f'{to_path}/y_data.txt', res,fmt='%d', delimiter=' ')
            #return res

    def prepare_datasets(self, compress_modes = [0,1,2,3,4]):
        self.compress_dataset(compress_modes)
        self.construct_trainX()
        self.construct_trainy()

def load_trainXy(X_file_path, y_file_path):
    train_X = np.loadtxt(X_file_path, delimiter=" ", dtype=float)
    train_y = np.loadtxt(y_file_path, delimiter=" ", dtype=float)
    return train_X, train_y
          
def test_accuracy_all(dp:Dataset_prep, test_size_proportions = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]):
    y_path = './training_data/y_data.txt'
    for compress_mode in range(5):
        X_path = f'./training_data/compressed_data_{dp.to_m}/{dp.modes[compress_mode]}/X_data.txt'
        result_path = f'./Test_results'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        X, y = load_trainXy(X_path, y_path)
        res = {}
        for ii in range(6):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_proportions[ii])
            res[test_size_proportions[ii]] = []
            for k_val in range(5,0,-1):
                cls1 = KNeighborsClassifier(k_val)
                cls1.fit(X_train, y_train)
                score = cls1.score(X_test, y_test)
                res[test_size_proportions[ii]].append(round(score,4))
        with open(f'{result_path}/m_{dp.to_m}_{dp.modes[compress_mode]}.txt') as f:
            f.write(str(res))
        f.close



if __name__ == '__main__':
    for ele in [50]:
        d1 = Dataset_prep(ele)
        d1.compress_dataset([i for i in range(5)])

#    ori_folder = '../data1'
#    folder_path = './cls_data'
#    dp = Dataset_prep()
#    dp.compress_dataset(ori_folder,300)
#    dp.filter_data(folder_path)
#    dp.construct_trainX(folder_path)
#    dp.construct_trainy(folder_path)
    
#    dp.load_trainX('./trainingdata1.txt')
