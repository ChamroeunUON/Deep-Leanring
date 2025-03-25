# 1. Import the library
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# 8. Evaluation model
# 8.1 true and predict data
predict_output = pd.read_csv("D:/MY LECTURE 2025/DL - Week 6/W6_Pred_ANN.csv")
actual_output = pd.read_csv("D:/MY LECTURE 2025/DL - Week 6/Actual_Output.csv")

y_pred = np.array(predict_output)
y_true = np.array(actual_output)


# axis = 0 (row), axis = 1 (col)
y_pred_mean = np.mean(y_pred, axis=1)
y_true_mean = np.mean(y_true, axis=1)

# 8.2 RMSE
RMSE = root_mean_squared_error(y_pred_mean,y_true_mean)
print("RMSE: ", "{:.2f}".format(RMSE))

# 8.3 MSE
MSE = mean_squared_error(y_pred_mean,y_true_mean)
print("MSE: ", "{:.2f}".format(MSE))

# 8.4 R
r2 = r2_score(y_pred_mean,y_true_mean)
R = np.sqrt(r2)
print("R: ", "{:.2f}".format(R))

# MAE
MAE = mean_absolute_error(y_pred_mean,y_true_mean)
print("MAE: ", "{:.2f}".format(MAE))

# 8.5 check pattern
plt.plot(y_pred_mean)
plt.plot(y_true_mean)
plt.title('predict vs actual')
plt.ylabel('angle (deg)')
plt.xlabel('Data Point (%)')
plt.legend(['predict', 'actual'], loc='upper right')
plt.show()




