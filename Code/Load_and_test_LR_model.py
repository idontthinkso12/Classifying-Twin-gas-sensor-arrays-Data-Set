import pickle
import numpy as np

filename = './saved_models/Logistic_reg_model1.sav'
model1 = pickle.load(open(filename,'rb'))

model_coeficients = model1.coef_
model_intercept = model1.intercept_

#print(model_coeficients.shape)
#print(model_intercept.shape)

#print(model_coeficients)

t = True
# test on whole data set

if t:
    X_test_path = './LR_data/compressed_data_10_Half/Non_Probability_Sampling/X_data.txt'
    y_test_path = './LR_data/y_data_10_Half.txt'

    test_X = np.loadtxt(X_test_path, delimiter=" ", dtype=float)
    test_y = np.loadtxt(y_test_path, delimiter=" ", dtype=float)

    print(model1.score(test_X, test_y))

