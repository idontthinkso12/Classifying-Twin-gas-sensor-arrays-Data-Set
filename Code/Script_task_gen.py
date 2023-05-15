import os

ms = [10,20,30,40]

with open('./Full_tmp.py','r') as f1:
    f1_lines = f1.readlines()
    
    for m in ms:
        to_file1 = f'./Full_{m}.py'
        f1_lines[1] = f'd1 = dp_yb.Dataset_prep({m})\n'
        with open(to_file1, 'w') as f2:
            for ls in f1_lines:
                f2.write(ls)
        f2.close()
    
    for m in ms:
        to_file2 = f'./Half_{m}.py'
        f1_lines[0] = f'import Dataset_prep_Nova_half as dp_yb\n'
        f1_lines[1] = f'd1 = dp_yb.Dataset_prep({m})\n'
        with open(to_file2, 'w') as f2:
            for ls in f1_lines:
                f2.write(ls)
        f2.close()
f1.close()

with open('./NOVA_Full_50.txt','r') as f1:
    f1_lines = f1.readlines()
    
    for m in ms:
        to_file1 = f'./NOVA_Full_{m}.txt'

        f1_lines[9] = f'#SBATCH --job-name="Full_{m}"\n'
        f1_lines[14] = f'#SBATCH --output="Full_{m}_output" # job standard output file (%j replaced by job id)\n'
        f1_lines[19] = f'python Full_{m}.py\n'
        
        with open(to_file1, 'w') as f2:
            for ls in f1_lines:
                f2.write(ls)
        f2.close()
    
    for m in ms:
        to_file2 = f'./NOVA_Half_{m}.txt'

        f1_lines[9] = f'#SBATCH --job-name="Half_{m}"\n'
        f1_lines[14] = f'#SBATCH --output="Half_{m}_output" # job standard output file (%j replaced by job id)\n'
        f1_lines[19] = f'python Half_{m}.py\n'

        with open(to_file2, 'w') as f2:
            for ls in f1_lines:
                f2.write(ls)
        f2.close()
f1.close()


