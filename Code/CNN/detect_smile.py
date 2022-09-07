import cv2
import numpy as np
import imutils
from tensorflow.keras.preprocessing.image import img_to_array # converting imgs to keras arrays
from tensorflow.keras.models import load_model # loading models from disk

# path of haar cascade compiler
cascade_path = 'CNN/haarcascade_frontalface_default.xml'
# path of lenet model
model_path = 'CNN/output/lenet.hdf5'

# load the face detector cascade and smile detector for CNN
detector = cv2.CascadeClassifier(cascade_path)
model = load_model(model_path)

# grab camera frames and set save directory
camera = cv2.VideoCapture(0)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
out = cv2.VideoWriter('CNN/output.mp4', fourcc, 20.0, (300, height))

while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	# resize the frame, convert it to grayscale, and then clone the original frame
	frame = imutils.resize(frame, width=300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frameClone = frame.copy()
    # detect faces in the input frame, then clone the frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # loop over the face bounding boxes
	for (fX, fY, fW, fH) in rects:
		# extract the ROI of the face from the grayscale image,
		roi = gray[fY:fY + fH, fX:fX + fW]
		# resize it to 28x28
		roi = cv2.resize(roi, (28, 28))
		roi = roi.astype("float") / 255.0
		# conver to keras array
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)
  		# determine the probabilities of both "smiling" and "not smiling" and set the label
		(notSmiling, smiling) = model.predict(roi)[0]
		label = "Smiling" if smiling > notSmiling else "Not Smiling"
    	# set the label and result box on the output frame
		cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
  	# display the detected result
	out.write(frame)
	cv2.imshow("Face", frameClone)
	# close when 'q' is pressed
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.release()
out.release()
cv2.destroyAllWindows()
