from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# assign temporary variables
no_of_examples = len(x_train)
split_ratio = 0.05
batch_size = 228
iters_per_epoch = no_of_examples*(1-split_ratio)/batch_size

# flatten 28*28 images to a 784 vector for each image
no_of_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], no_of_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], no_of_pixels).astype('float32')

# normalize data
x_train /= 255
x_test /= 255

# one-hot-encoding of output
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# model creation
model = Sequential()
model.add(Dense(no_of_pixels, input_dim=no_of_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

# model compilation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# use of validation set to control number of iterations (epochs) done.
earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

# model fitting
hist = model.fit(x_train, y_train, validation_split=0.05, epochs=16, callbacks=[earlyStopping], batch_size=batch_size, verbose=2)

# output score and number of iterations done
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
print("Number of iterations: ",  iters_per_epoch * len(hist.history['val_loss']))
