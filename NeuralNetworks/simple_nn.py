#Dependencies
import numpy as np
import pandas as pd#dataset import
import SEALPython.setup
dataset = pd.read_csv(r'C:\Users\saqla\Documents\Uni\Fourth Year\FYP\recommendation algorithm\dataset\phones\train.csv') #You need to change #directory accordingly
dataset.head(10) #Return 10 rows of data
from keras.constraints import min_max_norm, non_neg

X = dataset.iloc[:,:20].values
y = dataset.iloc[:,20:21].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

from keras.models import Sequential
from keras.layers import Dense# Neural network
model = Sequential()
# model.add(Dense(16, input_dim=20, activation='relu', kernel_constraint=non_neg(), use_bias=False))
# model.add(Dense(12, activation='relu', kernel_constraint=non_neg()))
model.add(Dense(4,input_dim=20, activation='softmax ', kernel_constraint=non_neg()))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit((X_train*100).astype('int64'), y_train,validation_data = (X_test*100,y_test), epochs=30, batch_size=64)

model.save('nn2.h5')

print(model.get_weights())