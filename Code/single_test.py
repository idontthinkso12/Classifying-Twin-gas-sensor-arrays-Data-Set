from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import Dataset_prep_yb as dp_yb
from time import time


X_data_path = './data/X_data.txt'
y_data_path = './data/y_data.txt'

X, y = dp_yb.load_trainXy(X_data_path, y_data_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


k_size = 1
cls1 = KNeighborsClassifier(k_size)

t1 = time()
for _ in range(10):    
    cls1.fit(X_train, y_train)
    score = cls1.score(X_test, y_test)
t2 = time()
print((t2-t1)/10)
#print(score)


#
# 0.03703022003173828    size of training data = 552
#
