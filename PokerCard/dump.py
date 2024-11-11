import numpy as np

def MAPE_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

y_true = np.array([0.3,0.3])

y_true = np.array([1.44,8.64,15.84,23.04])
y_pred = np.array([2.885, 8.632, 15.646, 23.164])


print ("MAPE_error", MAPE_error(y_true, y_pred))
