import Demo_yb as dm_yb

dm1 = dm_yb.Demo()
file_paths = ['./dummy/B5_GMe_F070_R1.txt']

dm1.demo(file_paths,
         sensor_wise =  False,
         mix_view =     False,
         time_col=      False,
         full_or_half= False,
         explicit= True
         )