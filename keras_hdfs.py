from keras.models import Sequential
from keras.layers import Dense
from keras.utils.io_utils import HDF5Matrix
from keras import metrics
import numpy as np

def create_dataset():
    import h5py
    X = np.random.randn(2000,10).astype('float32')
    y = np.random.randint(0, 2, size=(2000,1))
    f = h5py.File('test.h5', 'w')
    # Creating dataset to store features
    X_dset = f.create_dataset('my_data', (2000,10), dtype='f')
    X_dset[:] = X
    # Creating dataset to store labels
    y_dset = f.create_dataset('my_labels', (2000,1), dtype='i')
    y_dset[:] = y
    f.close()

create_dataset()

# Instantiating HDF5Matrix for the training set, which is a slice of the first 150 elements
X_train = HDF5Matrix('test.h5', 'my_data', start=0, end=1250)
y_train = HDF5Matrix('test.h5', 'my_labels', start=0, end=1250)

# Likewise for the test set
X_validation = HDF5Matrix('test.h5', 'my_data', start=1250, end=1500)
y_validation = HDF5Matrix('test.h5', 'my_labels', start=1250, end=1500)


# Likewise for the test set
X_test = HDF5Matrix('test.h5', 'my_data', start=1500, end=2000)
y_test = HDF5Matrix('test.h5', 'my_labels', start=1500, end=2000)

# HDF5Matrix behave more or less like Numpy matrices with regards to indexing
#print(y_train[10])
# But they do not support negative indices, so don't try print(X_train[-1])

model = Sequential()
model.add(Dense(64, input_shape=(10,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd')

# Note: you have to use shuffle='batch' or False with HDF5Matrix
model.fit(X_train, y_train, batch_size=32, shuffle='batch')

print("")

print(model.evaluate(X_validation, y_validation, batch_size=32))

y_pred = model.predict(X_test)

accuracy = metrics.binary_accuracy(y_test,y_pred)

print ("accuracy is ", accuracy)