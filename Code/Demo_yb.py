import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Demo:
    def __init__(self) -> None:
        self.col_names = ['t', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
        self.filename_map = {'B1':'Unit 1', 'B2':'Unit 2', 'B3':'Unit 3', 'B4':'Unit 4','B5':'Unit 5',
                             'GCO':'CO', 'GEa':'Ethanol', 'GEy':'Ethylene', 'GMe':'Methane', 
                             'F010': 0, 'F020': 1, 'F030': 2, 'F040': 3, 'F050': 4, 
                             'F060': 5, 'F070': 6, 'F080': 7, 'F090': 8, 'F100': 9, 
                             'R1': 'Repetition 1','R2': 'Repetition 2','R3': 'Repetition 3','R4': 'Repetition 4'}
        
        self.concentration_map = {'CO':[25.0, 50.0, 75.0, 100.0 , 125.0 ,150.0, 175.0, 200.0, 225.0 , 250.0],
                                  'Ethanol':[12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0 , 112.5, 125.0],
                                  'Ethylene':[12.5, 25, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0 , 112.5, 125.0],
                                  'Methane':[25.0, 50.0, 75.0, 100.0 , 125.0 ,150.0, 175.0, 200.0, 225.0 , 250.0]}
    
    def exp_info(self, file_name, short = False):
        d1, d2, d3, d4 = file_name[:-4].split('_')
        if short:
            return [d1, d2, d3, d4]
        o1 = self.filename_map[d1]
        o2 = self.filename_map[d2]
        o4 = self.filename_map[d4]
        idx = self.filename_map[d3]
        o3 = self.concentration_map[o2][idx]
        return [o1, o2, o3, o4]

    def demo(self, file_paths, sensor_wise = False, mix_view = False, time_col = True, full_or_half = True, explicit = False):    
        if full_or_half:
            max_t = 500
        else:
            max_t = 250

        if mix_view:
            mix_list = []
            for i in range(len(file_paths)):
                file_name = file_paths[i].split('/')[-1]
                unit_idx, gas_type, concen, rep_idx = self.exp_info(file_name,True)
                cur_col_names = self.col_names.copy()
                for ii in range(1, len(cur_col_names)):
                    cur_col_names[ii] += f'_{unit_idx}_{gas_type}_{concen}_{rep_idx}'
                data = pd.read_csv(file_paths[i], sep='\t', header=None, names=cur_col_names)
                if i == 0:
                    if time_col:
                        if explicit:
                            p = data.plot(x='t',subplots=sensor_wise, style='.-')
                        else:
                            p = data.plot(x='t',subplots=sensor_wise)
                    else:
                        t_column = np.arange(0,max_t,max_t/data.shape[0]).T                    
                        data['time'] = t_column
                        if explicit:
                            p = data.plot(x='time',subplots=sensor_wise, style='.-')
                        else:
                            p = data.plot(x='time',subplots=sensor_wise)
                else:
                    if time_col:
                        if explicit:
                            data.plot(ax=p, x='t',subplots=sensor_wise, style='.-')
                        else:
                            data.plot(ax=p, x='t',subplots=sensor_wise)
                    else:
                        t_column = np.arange(0,max_t,max_t/data.shape[0]).T                    
                        data['time'] = t_column
                        if explicit:
                            data.plot(ax=p, x='time',subplots=sensor_wise, style='.-')
                        else:
                            data.plot(ax=p, x='time',subplots=sensor_wise)
            plt.show()
        else:
            for i in range(len(file_paths)):
                data = pd.read_csv(file_paths[i], sep='\t', header=None, names=self.col_names)
                file_name = file_paths[i].split('/')[-1]
                unit_idx, gas_type, concen, rep_idx = self.exp_info(file_name)
                if time_col:
                    if explicit:
                        data.plot(x='t',title=f'{unit_idx, gas_type, concen, rep_idx}',subplots=sensor_wise, style='.-')
                    else:
                        data.plot(x='t',title=f'{unit_idx, gas_type, concen, rep_idx}',subplots=sensor_wise)
                else:
                    t_column = np.arange(0,max_t,max_t/data.shape[0]).T                    
                    data['time'] = t_column
                    if explicit:
                        data.plot(x='time',title=f'{unit_idx, gas_type, concen, rep_idx}',subplots=sensor_wise, style='.-')
                    else:
                        data.plot(x='time',title=f'{unit_idx, gas_type, concen, rep_idx}',subplots=sensor_wise)

            plt.show()