from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import Dataset_prep_yb as dp_yb



X_data_path = './data/X_data.txt'
y_data_path = './data/y_data.txt'

X, y = dp_yb.load_trainXy(X_data_path, y_data_path)

test_size_proportions = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
res = {}
for i in range(6):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_proportions[i])
    res[test_size_proportions[i]] = []
    for k_val in range(5,0,-1):
        cls1 = KNeighborsClassifier(k_val)
        cls1.fit(X_train, y_train)
        score = cls1.score(X_test, y_test)
        res[test_size_proportions[i]].append(round(score,4))

print(res)

# {0.6: [0.7804878048780488, 0.7994579945799458, 0.8130081300813008, 0.8509485094850948, 0.9024390243902439], 
#  0.5: [0.8990228013029316, 0.8892508143322475, 0.9055374592833876, 0.9022801302931596, 0.9348534201954397], 
#  0.4: [0.9227642276422764, 0.8943089430894309, 0.9146341463414634, 0.9186991869918699, 0.959349593495935], 
#  0.3: [0.8864864864864865, 0.8972972972972973, 0.9243243243243243, 0.9135135135135135, 0.9621621621621622], 
#  0.2: [0.9186991869918699, 0.9105691056910569, 0.926829268292683, 0.9349593495934959, 0.967479674796748], 
#  0.1: [0.9193548387096774, 0.9032258064516129, 0.9354838709677419, 0.9032258064516129, 0.9516129032258065]}        