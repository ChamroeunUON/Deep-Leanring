# 1. Import the library
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import root_mean_squared_error

# 2. Load data
# 2.1 Target Data
target_Train = pd.read_csv("Target_data.csv")

# 2.2 Convert data to array
target_Train_arr = np.array(target_Train)

# 2.3 Transpose target data
target_Train_tran = np.transpose(target_Train_arr)

# 3.1 Load Test data
Test_data = pd.read_csv("Input_Testin_Data.csv")
# 3.2 Convert data to array
Test_data_arr = np.array(Test_data)
# 3.3 Transpose target data
Test_data_Tra = np.transpose(Test_data_arr)

# 3.4 Normalize the data
# 3.4.1 formula to normalize
def norm(x):
    return (x - x.min()) / (x.max() - x.min())

Test_data_norm = norm(Test_data_Tra)
# 4. Recreate the exact same model, including its weights and the optimizer
ann_Model = tf.keras.models.load_model('w6_ANN_1.h5')

# 5. Prediction the test data with models
pre_Ann = ann_Model.predict(Test_data_norm)

# 6. Un-Label
res_Ann = pre_Ann + target_Train_tran.min()

# 7. Save data to execl field

df1 = pd.DataFrame(np.transpose(res_Ann))
df1.to_csv('W6_Pred_ANN.csv', index=None)

