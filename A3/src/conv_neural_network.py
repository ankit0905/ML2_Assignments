import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import backend as K

# sets the dimension ordering for image => 'th': (channels, rows, cols)
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train
y_train = y_train

# assign temporary variables
no_of_examples = len(x_train)
split_ratio = 0.05
batch_size = 228
iters_per_epoch = no_of_examples*(1-split_ratio)/batch_size

# flatten 28*28 images to a 784 vector for each image
print(x_train.shape)
no_of_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

# normalize data
x_train /= 255
x_test /= 255

# one-hot-encoding of output
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# model creation
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(1, 28, 28)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# use of validation set to control number of iterations (epochs) done.
earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=0,
                              verbose=0,
                              mode='auto')
# model fitting
hist = model.fit(x_train, y_train, 
                 validation_split=split_ratio,
                 epochs=16,
                 callbacks=[earlyStopping],
                 batch_size=batch_size,
                 verbose=1)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
print("Accuracy: ", scores[1])
print("Number of iterations: ",  iters_per_epoch * len(hist.history['val_loss']))
