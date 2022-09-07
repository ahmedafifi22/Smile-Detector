# Import packages
import numpy as np
import cv2
import dlib
import random
import os
import joblib # for svaing the svm  model 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures # Import polynomial regression and standardization 
import tqdm # progress bars
from sklearn.svm import SVC # Import svm
from sklearn.svm import LinearSVC # Import linear svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix # Extract 


# path to data set
folder_path='hog+svm/genki4k/'
# Labels file 
label='labels.txt'
# Pre-processed file path 
pic_folder='files1/'

# re-trained HOG + Linear SVM face detector included in the dlib library
detector = dlib.get_frontal_face_detector()
# trained face 68 Feature point detector
predictor = dlib.shape_predictor('hog+svm/shape_predictor_68_face_landmarks.dat')

# preprocessing input image and extracting mouth
def extract_mouth(img,detector,predictor):
    # Intercept the face
    #img_resize = cv2.resize(img,[200,200])
    img_gry=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rects = detector(img_gry, 0)
    if len(rects)!=0:
        mouth_x=0
        mouth_y=0
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[0]).parts()])
        # approx mouth range 
        for i in range(47,67):
            mouth_x+=landmarks[i][0,0]
            mouth_y+=landmarks[i][0,1]
        mouth_x=int(mouth_x/20)
        mouth_y=int(mouth_y/20)
        # Cut the picture
        img_cut=img_gry[mouth_y-20:mouth_y+20,mouth_x-20:mouth_x+20]
        return img_cut
    else:
        return 0 # if no face, return 0

#Extract eigenvalues
def get_feature(files_train,face,face_feature):
    for i in tqdm.tqdm(range(len(files_train))):
        img=cv2.imread(folder_path+pic_folder+files_train[i])
        cut_img=extract_mouth(img,detector,predictor)
        if type(cut_img)!=int:
            face.append(True)
            # resize input image to 64x64 for compatability
            cut_img=cv2.resize(cut_img,(64,64))
            # Boundary treatment padding
            padding=(8,8)
            winstride=(16,16)
            # compute hog
            hogdescrip=hog.compute(cut_img,winstride,padding).reshape((-1,))
            face_feature.append(hogdescrip)
        else:
            # append 0 if no face is detected
            face.append(False)
            face_feature.append(0)
            
# Check if face can be detected and extract (image,label)
# Remove the features of the image that cannot detect the face, and return the feature array and the corresponding labels 
def filter_face(face,face_feature,face_site):
    face_features=[]
    # Get tag
    label_flag=[]
    with open(folder_path+label,'r') as f:
        lines=f.read().splitlines()
    # Screen out faces that can detect , And collect the corresponding label
    for i in tqdm.tqdm(range(len(face_site))):
        # if face is detected
        if face[i]:
            # pop then delete the current element , The following elements should also move forward
            face_features.append(face_feature.pop(0))
            label_flag.append(int(lines[face_site[i]][0])) 
        else:
            face_feature.pop(0)
    datax=np.float64(face_features)
    datay=np.array(label_flag)
    #print (datax,datay)
    return datax,datay

# Polynomial SVM
def PolynomialSVC(degree,c=10):
    return Pipeline([
            # Transfer source data Mapping to 3 Order polynomial 
            ("poly_features", PolynomialFeatures(degree=degree)),
            # Standardization
            ("scaler", StandardScaler()),
            # SVC Linear classifier 
            ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42,max_iter=10000))
        ])
    

# SVM Gaussian kernel 
def RBFKernelSVC(gamma=1.0):
    return Pipeline([
        ('std_scaler',StandardScaler()),
        ('svc',SVC(kernel='rbf',gamma=gamma))
    ])

# Training the SVM
def train(files_train,train_site):
    train_face=[]
    # Feature array of human face 
    train_feature=[]
    # Extract the feature array of the training set 
    get_feature(files_train,train_face,train_feature)
    # Filter out the feature array of undetectable faces and return the imgs with labels
    train_x,train_y=filter_face(train_face,train_feature,train_site)
    svc=PolynomialSVC(degree=1)
    # fit the hyperplane
    svc.fit(train_x,train_y)
    return svc

# Testing model
def test(files_test,test_site,svc): 
    test_face=[]
    test_feature=[]
    get_feature(files_test,test_face,test_feature)
    test_x,test_y=filter_face(test_face,test_feature,test_site)
    pre_y=svc.predict(test_x)
    ac_rate=0
    # calculate accuracy
    for i in range(len(pre_y)):
        if(pre_y[i]==test_y[i]):
            ac_rate+=1
    ac=ac_rate/len(pre_y)*100
    #print("Accuracy rate is "+str(ac)+"%")
    return ac

# Calculating testing result and confusion matrix
def test1(files_test,test_site,svc):
    test_face = []
    test_feature = []
    get_feature(files_test,test_face,test_feature)
    test_x,test_y = filter_face(test_face,test_feature,test_site)
    # SVM results
    predict_y = svc.predict(test_x)
    # calculate and print confusion matrix
    conf_matrix = confusion_matrix(test_y, predict_y, labels=[0, 1])
    print('\nConfusion Matrix = \n',conf_matrix,'\n')
    tp=0
    tn=0
    # finding abs accuracy
    for i in range(len(predict_y)):
        if predict_y[i]==test_y[i] and predict_y[i]==1:
            tp+=1
        elif predict_y[i]==test_y[i] and predict_y[i]==0:
            tn+=1
    accuracy=2*tp/(tp+len(predict_y)-tn)
    print('Accuracy = ',accuracy*100,'%')

# Hog Parameters for feature extraction
winsize=(64,64)
blocksize=(32,32)
blockstride=(16,16)
cellsize=(8,8)
nbin=9
# apply hog descriptor
hog=cv2.HOGDescriptor(winsize,blocksize,blockstride,cellsize,nbin)

# extract files in the images folder
files=os.listdir(folder_path+pic_folder)
# after face extraction, 3878 pics remain
site=[i for i in range(3877)]
# grab a random sample of 3500 images for training
train_site=random.sample(site,3500)
# grab the remaining 378 images for testing
test_site=[]
for i in range(len(site)):
    if site[i] not in train_site:
        test_site.append(site[i])
files_train=[]

# Training set
for i in range(len(train_site)):
    files_train.append(files[train_site[i]])
    
# Testing set 
files_test=[]
for i in range(len(test_site)):
    files_test.append(files[test_site[i]])
#print(files_test)

# Train and save model
#svc = train(files_train,train_site)
save_path = 'hog+svm/genki4k/model2.pkl'
#joblib.dump(svc,save_path)

# load model and calc accuracy
load_svc=joblib.load(save_path)
#ac = test(files_test,test_site,load_svc)
#test1(files_test,test_site,load_svc)

# use svm model to detect smile
def smile_detector(img,svc):
    cut_img=extract_mouth(img,detector,predictor)
    a=[]

    if type(cut_img)!=int:
        cut_img=cv2.resize(cut_img,(64,64))
        #padding: Boundary treatment padding
        padding=(8,8)
        winstride=(16,16)
        hogdescrip=hog.compute(cut_img,winstride,padding).reshape((-1,))
        a.append(hogdescrip)
        result=svc.predict(a)
        a=np.array(a)
        return result[0]
    else :
        return 2

# Real-time detection
 # Turn on the camera
camera = cv2.VideoCapture(0)
ok=True
flag=0
# while camera is ON
while ok:
    ok,img = camera.read()
     # Convert the image to grayscale  and detect smile
    result=smile_detector(img,load_svc)
    # if 1 : smiling
    if result==1:
        img=cv2.putText(img,'Smiling',(21,50),cv2.FONT_HERSHEY_COMPLEX,2.0,(0,0,0),2)
    # if 0 : not smiling
    elif result==0:
        img=cv2.putText(img,'Not Smiling',(21,50),cv2.FONT_HERSHEY_COMPLEX,2.0,(0,0,0),2)
    # if no face is detected
    else:
        img=cv2.putText(img,'No Face Detected',(21,50),cv2.FONT_HERSHEY_COMPLEX,2.0,(0,0,0),2)
    cv2.imshow('video', img)
    k = cv2.waitKey(1)
    # if 'ESC' is pressed, quit
    if k == 27:
        break
    # if 's' is pressed save a capture of the video stream
    elif k==115:
        pic_save_path='hog+svm/genki4k/saved_results/'+str(flag)+'.jpg'
        flag+=1
        cv2.imwrite(pic_save_path,img)
camera.release()
cv2.destroyAllWindows()

