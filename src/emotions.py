# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 12:19:03 2020

@author: hp
"""




import dlib
import cv2
import numpy as np
import glob
import random
import math
import itertools
from sklearn.svm import SVC
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix
import pickle






path='D:/Work-Space/S3/ImageMining/ckplus/CK+48'

#Emotion list
emotions = ["anger","contempt", "disgust","fear", "happy", "sadness", "surprise"]
detector = dlib.get_frontal_face_detector()
model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clf = SVC(C=0.01, kernel='linear', decision_function_shape='ovo', probability=True)   #Set the classifier as a support vector machines with linear kernel


def get_files(emotion):
    images = glob.glob(path+"\\%s\\*" %emotion)
    random.shuffle(images)
    training_set = images[:int(len(images)*0.8)]   #get 80% of image files to be trained
    testing_set = images[-int(len(images)*0.2):]   #get 20% of image files to be tested
    return training_set, testing_set




def get_landmarks(image):
    detections = detector(image, 1)
    #For all detected face instances individually
    for k,d in enumerate(detections):
        #get facial landmarks with prediction model
        shape = model(image, d)
        xpoint = []
        ypoint = []
        for i in range(17, 68):
            xpoint.append(float(shape.part(i).x))
            ypoint.append(float(shape.part(i).y))

        #center points of both axis
        xcenter = np.mean(xpoint)
        ycenter = np.mean(ypoint)
        #Calculate distance between particular points and center point
        xdistcent = [(x-xcenter) for x in xpoint]
        ydistcent = [(y-ycenter) for y in ypoint]

        #prevent divided by 0 value
        if xpoint[11] == xpoint[14]:
            angle_nose = 0
        else:
            #point 14 is the tip of the nose, point 11 is the top of the nose brigde
            angle_nose = int(math.atan((ypoint[11]-ypoint[14])/(xpoint[11]-xpoint[14]))*180/math.pi)

        #Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
        if angle_nose < 0:
            angle_nose += 90
        else:
            angle_nose -= 90

        landmarks = []
        for cx,cy,x,y in zip(xdistcent, ydistcent, xpoint, ypoint):
            #Add the coordinates relative to the centre of gravity
            landmarks.append(cx)
            landmarks.append(cy)

            #Get the euclidean distance between each point and the centre point (the vector length)
            meanar = np.asarray((ycenter,xcenter))
            centpar = np.asarray((y,x))
            dist = np.linalg.norm(centpar-meanar)

            #Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
            if x == xcenter:
                angle_relative = 0
            else:
                angle_relative = (math.atan(float(y-ycenter)/(x-xcenter))*180/math.pi) - angle_nose
                #print(anglerelative)
            landmarks.append(dist)
            landmarks.append(angle_relative)

    if len(detections) < 1:
        #In case no case selected, print "error" values
        landmarks = "error"
    return landmarks


def make_sets():
    training_data = []
    training_label = []
    testing_data = []
    testing_label = []
    for emotion in emotions:
        training_set, testing_set = get_files(emotion)
        #add data to training and testing dataset, and generate labels 0-4
        for item in training_set:
            #read image
            img = cv2.imread(item)
            #convert to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe_img = clahe.apply(gray_img)
            landmarks_vec = get_landmarks(clahe_img)

            if landmarks_vec == "error":
                pass
            else:
                training_data.append(landmarks_vec)
                training_label.append(emotions.index(emotion))

        for item in testing_set:
            img = cv2.imread(item)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe_img = clahe.apply(gray_img)
            landmarks_vec = get_landmarks(clahe_img)
            if landmarks_vec == "error":
                pass
            else:
                testing_data.append(landmarks_vec)
                testing_label.append(emotions.index(emotion))

    return training_data, training_label, testing_data, testing_label
def create_model():
    accur_lin = []
    max_accur = 0
    for i in range(0,1):
        #Make sets by random sampling 80/20%
        print("Marking set %s" %i)
        X_train, y_train, X_test, y_test = make_sets()

        #Turn the training set into a numpy array for the classifier
        np_X_train = np.array(X_train)
        np_y_train = np.array(y_train)
        #train SVM
        print("Training SVM Classifier %s" %i)
        clf.fit(np_X_train, np_y_train)

        np_X_test = np.array(X_test)
        np_y_test = np.array(y_test)
        #Use score() function to get accuracy
        print("Getting accuracy score -- %s" %i)
        #npar_pred = np.array(X_test)
        pred_lin = clf.score(np_X_test, np_y_test)
        #y_pred = clf.predict(X_test)
        #Find Best Accuracy and save to file
        if pred_lin > max_accur:
            max_accur = pred_lin
            max_clf = clf
            X_test_opt = np_X_test
            y_test_opt = np_y_test
            X_train_opt = np_X_train
            y_train_opt = np_y_train
            test_pred = max_clf.predict(np_X_test)
            #train_pred = max_clf.predict(np_X_train)
            #print("Hello")
        #y_pred = clf.predict(np_X_test)
        print("Test Accuracy: ", pred_lin)
        #print(confusion_matrix(np_y_test, y_pred))
        accur_lin.append(pred_lin)  #Store accuracy in a list

    print("Mean Accuracy Value: %.3f" %np.mean(accur_lin))   #Get mean accuracy of the 10 runs
    #test_pred = max_clf.predict(X_test_opt)
    #print(confusion_matrix(y_train_opt, train_pred))
    #print(classification_report(y_train_opt, train_pred))
    print(confusion_matrix(y_test_opt, test_pred))
    print(classification_report(y_test_opt, test_pred))

    return max_accur, max_clf

if __name__ == '__main__':
    max_accur, max_clf = create_model()
    print('Best accuracy = ', max_accur*100, 'percent')
    print(max_clf)
    try:
        os.remove('D:\Work-Space\S3\ImageMining\models\model1.pkl')
    except OSError:
        pass
    output = open('D:\Work-Space\S3\ImageMining\models\model1.pkl', 'wb')
    pickle.dump(max_clf, output)
    output.close()
    
    
    
    
    
    
mod=pickle.load(open("D:\Work-Space\S3\ImageMining\models\model1.pkl", "rb"))
training_data = []
image = cv2.imread("D:\\Work-Space\\S3\\ImageMining\\images\\train\\angry\\22.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(gray)

#Get Point and Landmarks
landmarks_vectorised = get_landmarks(clahe_image)

#print(landmarks_vectorised)

    #Predict emotion
training_data.append(landmarks_vectorised)
npar_pd = np.array(training_data)

prediction_emo = mod.predict(npar_pd)
    
prediction_emo = prediction_emo[0]
print(emotions[prediction_emo])

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
