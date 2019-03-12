# importing all the necessary packages for the project
import numpy as np
import cv2
import os
import pickle
import argparse
from imutils import paths
import imutils

# constructing the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i","--dataset",default='dataset',help="path to input image")
ap.add_argument("-e","--embeddings",default='output/embeddings.pickle',help="path to output serialized db of facial embeddings")
ap.add_argument("-d","--detector",default='face_detection_model',help="path to openCV's deep learning detector")
ap.add_argument("-m","--structure_model",default='openface_nn4.small2.v1.t7',help="path to openCV's deep learning structure model")
ap.add_argument("-c","--confidence",type=float, default=0.5,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load face detector from disk
print("[INFO] loading face detector..")
protoPath = os.path.sep.join([args['detector'],'deploy.prototxt'])
modelPath = os.path.sep.join([args['detector'],'res10_300x300_ssd_iter_140000.caffemodel'])
detector = cv2.dnn.readNetFromCaffe(protoPath,modelPath)

# load face structure model from disk
print("[INFO] loading face recognizer..")
embedder = cv2.dnn.readNetFromTorch(args['structure_model'])

#getting the paths of input images
print("[INFO] quantifying faces..")
imagePaths = list(paths.list_images(args['dataset']))

knownEmbeddings = []
knownNames = []

#initializing total number of faces processed
total = 0

#loop through all the image paths
for (i,imagePath) in enumerate(imagePaths):
    # extract the person name from image path
    print("[INFO] processing image{}/{}".format(i+1,len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    #resizing the image
    image = cv2.imread(imagePath)
    image = imutils.resize(image,width=600)
    (h,w) = image.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0,177.0,123.0),swapRB=False,crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0,0,:,2])
        confidence = detections[0,0,i,2]

        if confidence > args['confidence']:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype('int')

            face = image[startY:endY,startX:endX]
            (fH,fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue
            faceBlob = cv2.dnn.blobFromImage(face,1.0/255,(96,96),(0,0,0),swapRB=True,crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total+=1

#dump the names and embeddings to disk
print("[INFO] serializing {} encodings".format(total))
data = {"embeddings" : knownEmbeddings,"names":knownNames}
f = open(args['embeddings'],"wb")
f.write(pickle.dumps(data))
f.close()
