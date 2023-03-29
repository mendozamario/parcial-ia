
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_excel('entrenamientomalaria.xlsx')

print(df.head())
df.isnull().sum()

scaler = StandardScaler()
x = df.drop(['excess_cases1'], axis=1)
y = df['excess_cases1']
x= scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(30,activation='relu',input_shape=(x_train.shape[1],)),
    keras.layers.Dense(60,activation='relu'),
    keras.layers.Dense(30,activation='relu'),
    keras.layers.Dense(1,activation='linear'),
])

model.compile(optimizer='adam', loss='mse', metrics=['mse',keras.metrics.RootMeanSquaredError()])
regularization = keras.regularizers.l2(0.01)
model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=regularization))
model.add(keras.layers.Dense(1, activation='linear', kernel_regularizer=regularization))
model.add(keras.layers.Dropout(0.5))

history = model.fit(x_train, y_train, epochs=100, validation_split=0.2)

mse,mae,rmse = model.evaluate(x_test, y_test, verbose=2)
print(f"MSE:{mse:.2f}")
print(f"MAE:{mae:.2f}")
print(f"RMSE:{rmse:.2f}")

x_test_scaled = scaler.transform(x_test)
y_pred = model.predict(x_test_scaled)
y_pred = y_pred.flatten()
y_pred_binary = (y_pred > 0.5).astype(int)

for i in range(len(y_pred_binary)):
    if y_pred_binary[i] >=0.5:
       print ("si habra brote de malaria")
else:
    print ("no habra brote de malaria")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()