# Importing libraries
import os
from zipfile import ZipFile
# import cv2
import imghdr
import keras
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np

# Extracting dataset (only once)
with ZipFile('happySad.zip', 'r') as zip:
    zip.extractall()
    print('done')

# removing weird images (only once)
ext = ['jpeg', 'jpg', 'bmp', 'png']
imageClasses = ['happy', 'sad']
for folder in imageClasses:
    for image in os.listdir(os.path.join('happySad', folder)):
        imagePath = os.path.join('happySad', folder, image)
        try:
            img = Image.open(imagePath)
            tip = imghdr.what(imagePath)
            if tip not in ext:
                print('Image not in ext {}'.format(imagePath))
                os.remove(imagePath)
        except Exception as e:
            print('Issue with image {}'.format(imagePath))

# Counting final images
print(len(os.listdir(os.path.join('happySad', 'happy'))))
print(len(os.listdir(os.path.join('happySad', 'sad'))))

# Creating pipline
data = tf.keras.utils.image_dataset_from_directory('happySad', batch_size=20, image_size=(256, 256))
data.class_names

# Image iterator
dataIterator = data.as_numpy_iterator()
batch = dataIterator.next()
batch[0].shape

# Plotting set of images
batch = dataIterator.next()
fig, ax = plt.subplots(ncols=4, figsize=(10, 10))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

data = data.map(lambda x, y: (x / 255, y))
print(data.as_numpy_iterator().next()[0].min(),
      '\n', data.as_numpy_iterator().next()[0].max())

# Dividing into train, val and test
batches = len(data)  # Number of different batches
trainSizeBatches = int(batches * .7)
valSizeBatches = int(batches * .2) + 1
testSizeBatches = int(batches * .1) + 1
trainSizeBatches + valSizeBatches + testSizeBatches

# Creating datasets
train = data.take(trainSizeBatches)
val = data.skip(trainSizeBatches).take(valSizeBatches)
test = data.skip(trainSizeBatches + valSizeBatches).take(testSizeBatches)
print("Number of batches for train:", len(train))

# Model architecture
model = Sequential()
model.add(tf.keras.layers.RandomFlip('horizontal'))
# model.add(tf.keras.layers.RandomZoom(0.2))
# model.add(tf.keras.layers.RandomRotation(0.2))
model.add(tf.keras.layers.Rescaling(1 / 255, offset=-1))
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.build(input_shape=(None, 256, 256, 3))
model.summary()

# Compiling model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Creating log file
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='log')

# Training model
hist = model.fit(train, epochs=45, validation_data=val, callbacks=[tensorboard_callback])

# Evaluating training performance
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# Evaluation objects
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

# Evaluating train
for batch in train.as_numpy_iterator():
    X, y = batch  # Extracting elements from a batch (each batch is composed by X array and Y labels)
    yhat = model.predict(X)
    pre.update_state(y, yhat)  # Precision
    re.update_state(y, yhat)  # Recall
    acc.update_state(y, yhat)

# Evaluating validation
for batch in val.as_numpy_iterator():
    X, y = batch  # Extracting elements from a batch (each batch is composed by X array and Y labels)
    yhat = model.predict(X)
    pre.update_state(y, yhat)  # Precision
    re.update_state(y, yhat)  # Recall
    acc.update_state(y, yhat)

# Evaluating test
testBatches = test.as_numpy_iterator().next()
print(testBatches[0].shape)
fig, ax = plt.subplots(ncols=4, figsize=(10, 10))
for idx, img in enumerate(testBatches[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(testBatches[1][idx])

for batch in test.as_numpy_iterator():
    X, y = batch  # Extracting elements from a batch (each batch is composed by X array and Y labels)
    yhat = model.predict(X)
    pre.update_state(y, yhat)  # Precision
    re.update_state(y, yhat)  # Recall
    acc.update_state(y, yhat)

print("Precision:", pre.result().numpy(),
      "\nRecall:", re.result().numpy(),
      "\nAccuracy:", acc.result().numpy())

# Evaluating new data
newTest = Image.open('jorgeFeliz2.jpg')
newTest = newTest.rotate(270)
imgplot = plt.imshow(newTest)
plt.show()

# Rescaling imagen (model trained with 255x255 images)
resized = tf.image.resize(np.array(newTest), (256, 256))
imgplot = plt.imshow((resized / 255))
plt.show()

# Predicting
yhat = model.predict(np.expand_dims(resized, 0))
print(yhat)
