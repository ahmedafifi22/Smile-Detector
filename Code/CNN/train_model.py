# importing the training packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
import os

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
            # initialize the model
		    model = Sequential()
		    inputShape = (height, width, depth)
		    # update the input shape
		    if K.image_data_format() == "channels_first":
			    inputShape = (depth, height, width)
   
            # CONV => RELU => POOL layers
		    model.add(Conv2D(20, (5, 5), padding="same",
			    input_shape=inputShape))
		    model.add(Activation("relu"))
		    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		    model.add(Conv2D(50, (5, 5), padding="same"))
		    model.add(Activation("relu"))
		    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            # FC => RELU layers
		    model.add(Flatten())
		    model.add(Dense(500))
		    model.add(Activation("relu"))
            # softmax classifier
		    model.add(Dense(classes))
		    model.add(Activation("softmax"))
		    # return the architecture
		    return model

dataset_path = 'CNN/pics'
model_path = 'CNN/output/lenet.hdf5'

data = []
labels = []

# loop over the input images
for imagePath in sorted(list(paths.list_images(dataset_path))):
	# load the imag
	image = cv2.imread(imagePath)
	# convert to gray scale and resize
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = imutils.resize(image, width=28)
	# convert to array
	image = img_to_array(image)
	data.append(image)
	# extract the class label and update
	label = imagePath.split(os.path.sep)[-3]
	label = "smiling" if label == "positives" else "not_smiling"
	labels.append(label)

# normalize the  pixel intensities to [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 2)

# calculate the total number of training images in each class
classTotals = labels.sum(axis=0)
# store the class weights in a dict
classWeight = dict()

# calculate the class weights by looping over all classes
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]
 
# split data into 80% for training and 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.2, stratify=labels, random_state=42)

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])

# train the CNN
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	class_weight=classWeight, batch_size=64, epochs=20, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# save the model
print("[INFO] serializing network...")
model.save(model_path)

# plot the training accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy/Loss")
plt.legend()
plt.show()

