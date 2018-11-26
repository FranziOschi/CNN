import keras
from keras.datasets import mnist
from keras import backend as k
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, AveragePooling2D

class CNN_architectures:

	def LeNet_5(self):
		#model building
		self.model = Sequential()

		self.model.add(Conv2D(filters = 6, kernel_size = 5, strides = 1, activation='tanh', input_shape = (32,32,1)))

		self.model.add(AveragePooling2D(pool_size= 2, strides = 2))

		self.model.add(Conv2D(filters = 16, kernel_size = 5, strides = 1, activation='tanh', input_shape = (14,14,6)))

		self.model.add(AveragePooling2D(pool_size= 2, strides = 2))

		self.model.add(Conv2D(filters = 120, kernel_size = 5, strides = 1, activation='tanh'))

		self.model.add(Flatten())

		self.model.add(Dense(units = 84, activation='tanh'))

		self.model.add(Dense(units = 10, activation = 'softmax'))
