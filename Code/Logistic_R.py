from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import Dataset_prep_yb as dp_yb
import pickle
from sklearn.preprocessing import StandardScaler

X_data_path = f'./dummy/X_data_four_point_square.txt'
y_data_path = './dummy/y_data_50_Half.txt'

X, y = dp_yb.load_trainXy(X_data_path, y_data_path)

scaler = StandardScaler()

test_size_proportions = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
res = {}
for i in range(6):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_proportions[i])
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    cls1 = LogisticRegression(solver='lbfgs', max_iter=2000)
    cls1.fit(X_train_scaled, y_train)
    score = cls1.score(X_test_scaled, y_test)
    res[test_size_proportions[i]] = round(score,4)

    #if i == 5:
    #    filename = './saved_models/Logistic_reg_model2.sav'
    #    pickle.dump(cls1, open(filename,'wb'))

print(res)