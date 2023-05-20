from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping



photos = np.load('photos_AI.npy')
labels = np.load('labels_AI.npy')
x_train, x_test, y_train, y_test = train_test_split(photos, labels, test_size=0.1)
print(x_train.shape)  #(10778,90,120,3)
print(x_test.shape)   #(1198,90,120,3)

x_train = x_train.reshape(10778, 90, 120, 3)
x_test = x_test.reshape(1198, 90, 120, 3)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

labels_vehicle = ['bicycle','boat','car','motorbike','airplane','train','truck']



y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7)

model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(90,120,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(7, activation='softmax'))

model.summary()

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=12, epochs=100, verbose=1)

test_loss,test_acc=model.evaluate(x_test,y_test)

print('test_acc:',test_acc)
print('test_loss:',test_loss)
