import dlib 
import numpy as np
import cv2
import os
 
# dlib face extractor 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# path of GENKI-4K data set
path_read = "hog+svm/genki4k/files"
num=0
for file_name in os.listdir(path_read):
	# path of the picture
    pic_path=(path_read +"/"+file_name)
    img=cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # extract image size
    img_shape=img.shape
    img_height=img_shape[0]
    img_width=img_shape[1]
   
    #save path
    path_save="hog+svm/genki4k/files1" 
    # dlib testing 
    dets = detector(img,1)
    print("Face a few:", len(dets))
    for k, d in enumerate(dets):
        if len(dets)>1:
            continue
        num=num+1
        # Calculate the size of the rectangle
        pos_start = tuple([d.left(), d.top()])
        pos_end = tuple([d.right(), d.bottom()])
 
        # Calculate the size of the rectangle 
        height = d.bottom()-d.top()
        width = d.right()-d.left()
 
        # Generate an empty image equal to the face size 
        img_blank = np.zeros((height, width, 3), np.uint8)
        for i in range(height):
            # To prevent cross-border 
            if d.top()+i>=img_height:
                continue
            for j in range(width):
                if d.left()+j>=img_width:
                    continue
                img_blank[i][j] = img[d.top()+i][d.left()+j]
        img_blank = cv2.resize(img_blank, (200, 200), interpolation=cv2.INTER_CUBIC)

        cv2.imencode('.jpg', img_blank)[1].tofile(path_save+"/"+file_name)