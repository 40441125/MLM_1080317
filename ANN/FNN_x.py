import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
digits = datasets.load_digits()
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
X_digits = digits.images.reshape((len(digits.images), -1))
Y = dataset[:,4]
Y_digits = digits.target


originY = Y
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
dummy_y_digits = np_utils.to_categorical(Y_digits)

train_data, test_data, train_label, _ = train_test_split(X, dummy_y, test_size=0.2,random_state=1)
_, _, _, test_label2 = train_test_split(X, encoded_Y, test_size=0.2,random_state=1)

train_data_digits, test_data_digits, train_label_digits, _ = train_test_split(X_digits, dummy_y_digits, test_size=0.2,random_state=1)
_, _, _, test_label2_digits = train_test_split(X_digits, Y_digits, test_size=0.2,random_state=1)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def baseline_model_digits():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=64, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model()

history = model.fit(train_data,train_label,validation_split=0.3,epochs=200,batch_size=5,verbose=1)

print(model.predict_classes(test_data))
print(test_label2)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


model = baseline_model_digits()
history_digits = model.fit(train_data_digits,train_label_digits,validation_split=0.3,epochs=200,batch_size=32,verbose=1)
print(model.predict_classes(test_data_digits)[:30])
print(test_label2_digits[:30])

plt.plot(history_digits.history['acc'])
plt.plot(history_digits.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/