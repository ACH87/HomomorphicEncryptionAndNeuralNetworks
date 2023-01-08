import numpy as np
import pandas as pd#dataset import
dataset = pd.read_csv(r'C:\Users\saqla\Documents\Uni\Fourth Year\FYP\recommendation algorithm\dataset\phones\train.csv') #You need to change #directory accordingly
dataset.head(10) #Return 10 rows of data

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
from keras_layers.Paillier_Dense import Paillier_Dense

from Security_Functions import PallierFunctions
pal = PallierFunctions(min=30, max=50)
test = np.where(X_test<0, 0, X_test)*100
print(test)
test = pal.encrypt_numpy(test[1].astype('int64'))
print(test)

model = Sequential()
# model.add(Paillier_Dense(16, n=pal.public_key['n'], input_dim=20, activation='relu'))
# model.add(Paillier_Dense(12, activation='relu', n=pal.public_key['n']))
model.add(Paillier_Dense(4, input_dim=20, n=pal.public_key['n']))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights(r'C:\Users\saqla\Documents\Uni\Fourth Year\FYP\recommendation algorithm\src\nn2.h5')

from keras.layers import  Dense
model2 = Sequential()
# model2.add(Dense(16, input_dim=20, activation='relu'))
# model2.add(Dense(12, activation='relu'))
model2.add(Dense(4, input_dim=20))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights(r'C:\Users\saqla\Documents\Uni\Fourth Year\FYP\recommendation algorithm\src\nn2.h5')

# model.fit(X_train, y_train, epochs=1, batch_size=10)

result= model.predict(np.reshape(test, [1,20]))

def softmax(z):
    """Compute softmax values for each sets of scores in x."""
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div

print(result)
print((pal.decrypt_numpy(result.astype('int64'))))
print(y_test[1])
print('result2', model2.predict(np.reshape(X_test[0], [1,20])*100))
print('softmax result', softmax(model2.predict(np.reshape(X_test[0]*100, [1,20]))))
