# 1. Import the library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop




#from tensorflow.python.keras.optimizers import RMSprop

#Dense = tf.keras.layers.Dense
#Sequential = tf.keras.models.Sequential
#RMSprop = tf.Keras.optimizers.RMSprop
#Dropout
#2. Load data
# 2.1 Input data
Input_data = pd.read_csv("Input_data.csv")
# 2.2 Target Data
Target_data = pd.read_csv("Target_data.csv")

# 3. Pre-processing data
# 3.1 Convert data to array
Input_arr = np.array(Input_data)
Target_arr = np.array(Target_data)

# 3.2 Transpose the matrix
Input_tranp = np.transpose(Input_arr)
Target_tranp = np.transpose(Target_arr)

# 4. Normalize the data
# 4.1 formula to normalize
def norm(x):
    return (x - x.min()) / (x.max() - x.min())

Input_norm = norm(Input_tranp)

# 5 Label the target output
# 5.1 formula to label the target output
def label(y):
    z = (y - y.min())
    return np.round(z)
Target_tranp = label(Target_tranp)

# 6 Split data 80% for train and 20% for validation
Input_trai, Input_valid, Target_trai, Target_valid = train_test_split(Input_norm, Target_tranp, test_size=0.2)

# 7. Create deep learning model
ann_Model_1 = Sequential([
        Dense(units=250, input_dim=Input_trai.shape[1], activation='sigmoid'),
        Dense(units=250, activation='relu'),
        Dense(Target_trai.shape[1], activation='elu')
    ])
ann_Model_2 = Sequential([
        Dense(units=250, input_dim=Input_trai.shape[1], activation='sigmoid'),
        Dropout(0.1),
        Dense(units=250, activation='relu'),
        Dropout(0.1),
        Dense(Target_trai.shape[1], activation='elu')
    ])
ann_Model_3 = Sequential()
ann_Model_3.add(Dense(units=250, input_dim=Input_trai.shape[1], activation='sigmoid'))
ann_Model_3.add(Dropout(0.1))
ann_Model_3.add(Dense(units=250, activation='relu'))
ann_Model_3.add(Dropout(0.1))
ann_Model_3.add(Dense(Target_trai.shape[1], activation='elu'))

# Check model property
print(ann_Model_1.summary())
print(ann_Model_2.summary())
print(ann_Model_3.summary())

# 8. compile and train model
ann_Model_1.compile(loss='mean_squared_error', optimizer=RMSprop(learning_rate=0.007), metrics=['accuracy'])
history = ann_Model_1.fit(Input_trai, Target_trai, validation_data=(Input_valid, Target_valid), batch_size=10, epochs=150, verbose = 2)

# 9. visualize loss in deep learning
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 10. visualize accuracy in deep learning
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 11. save the models
ann_Model_1.save('w5_ImplementationANN_1.h5')

# 12. check the performance of model with actual output and predict output

