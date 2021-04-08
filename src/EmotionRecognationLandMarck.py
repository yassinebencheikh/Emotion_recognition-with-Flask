# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 10:08:43 2020

@author: Boulaalam
"""

# import the necessary packages
from imutils import paths
from skimage import feature
import numpy as np
import cv2
import glob
import random
import math
import dlib
import pandas as pd
from sklearn.svm import SVC



clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("E:/Master/MS3_2019_2020/Image mining/miniProjet/emotinrecognition/shape_predictor_68_face_landmarks/shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
data = {} #Make dictionary for all values
#data['landmarks_vectorised'] = []



def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"
        

        
def make_sets(directory,emotions):
    training_data = []
    training_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        files = glob.glob(directory+"\\%s\\*" %emotion)
        random.shuffle(files)
        training = files[:int(len(files))] 
        
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(emotions.index(emotion))

    return np.array(training_data), np.array(training_labels)



# directory of all class images

#directory='E:/Master/MS3_2019_2020/Image mining/miniProjet/emotinrecognition/yassin/test'    
directory='E:\\Master\\MS3_2019_2020\\Image mining\\miniProjet\\emotinrecognition\\coco\\Training'
# emotions vector 
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] #Emotion list


######################## Random Forest  #####################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
forest_model = RandomForestClassifier(random_state=2,n_estimators=100)

######################### Split train+test #######################################
from sklearn.model_selection import train_test_split
# landmarck  function
X, Y = make_sets(directory,emotions)


dataFrame=pd.DataFrame(X) 
dataFrame[dataFrame.shape[1]]=Y

# shuffling
from sklearn.utils import shuffle


######################## Random Forest  #####################################

accur_lin = []
for i in range(0,10):
    print("Making sets %s" %i)
    dataFrame = shuffle(dataFrame)
    X=dataFrame.iloc[:,0:-1].values
    Y=dataFrame.iloc[:,-1].values
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10,random_state=2)# 80% training and 30% test
    # Random Forest model
    forest_model.fit(X_train, y_train)
    y_preds = forest_model.predict(X_test)
    accu_score=accuracy_score(y_test,y_preds)
    print("RFC accueacy : ",accu_score)
    accur_lin.append(accu_score) #Store accuracy in a list
print("Mean value RFC: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs



######################## Decision Tree  #####################################
# Create Decision Tree classifer object
from sklearn.tree import DecisionTreeClassifier 
DTC = DecisionTreeClassifier()

accur_DTC = []
for i in range(0,10):
    print("Making sets %s" %i)
    dataFrame = shuffle(dataFrame)
    X=dataFrame.iloc[:,0:-1].values
    Y=dataFrame.iloc[:,-1].values
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10,random_state=2)# 80% training and 30% test
    # Decision Tree model
    DTC.fit(X_train, y_train)
    y_preds = DTC.predict(X_test)
    accu_score=accuracy_score(y_test,y_preds)
    print("DTC accueacy : ",accu_score)
    accur_DTC.append(accu_score) #Store accuracy in a list
print("Mean value  Decision Tree : %s" %np.mean(accur_DTC)) #FGet mean accuracy of the 10 runs


######################## SVM #####################################

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

accur_lin = []
for i in range(0,10):
    print("Making sets %s" %i)
    dataFrame = shuffle(dataFrame)
    X=dataFrame.iloc[:,0:-1].values
    Y=dataFrame.iloc[:,-1].values
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10,random_state=2)# 80% training and 30% test
    # Random Forest model
    svclassifier.fit(X_train, y_train)
    y_preds = svclassifier.predict(X_test)
    accu_score=accuracy_score(y_test,y_preds)
    print("SVM accueacy : ",accu_score)
    accur_lin.append(accu_score) #Store accuracy in a list
print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs






## features extraction 
#X, Y = make_sets(directory,emotions)
#X_data=X
#Y_data=Y


## Import label encoder 
#from sklearn import preprocessing 
## label_encoder object knows how to understand word labels. 
#label_encoder = preprocessing.LabelEncoder()
#Y=label_encoder.fit_transform(Y)



X=dataFrame.iloc[:,0:-1].values
Y=dataFrame.iloc[:,-1].values


########################## Standarisation #######################################
from sklearn.preprocessing import StandardScaler as SS
ssX= SS()
X=ssX.fit_transform(X)


############################ ACP ##############################################


#classe pour l'ACP
from sklearn.decomposition import PCA
#instanciation
acp = PCA(svd_solver='full')
#calculs
X_acp = acp.fit_transform(X)

#nombre de composantes calculées
print(acp.n_components_) 

#variance expliquée
print(acp.explained_variance_)

#proportion de variance expliquée
print(acp.explained_variance_ratio_)

#nombre d'observations
n = X.shape[0]
#nombre de variables
p = X.shape[1]


#valeur corrigée
eigval = (n-1)/n*acp.explained_variance_
print(eigval)

#scree plot
import matplotlib.pyplot as plt
plt.plot(np.arange(1,p+1),eigval)
plt.title("Scree plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.show()


####################################################################
# N_explained is the number of the columns which contains 
# most of the information in the table explained_variance_N_explaineed
N_explained=50
Z_acp=X_acp[:,0:N_explained]
#nombre d'observations
n_acp = Z_acp.shape[0]
#nombre de variables
p_acp = Z_acp.shape[1]


#valeur corrigée
eigval = (n_acp-1)/n_acp*acp.explained_variance_[0:N_explained]
print(eigval)
#scree plot
plt.plot(np.arange(1,p_acp+1),eigval)
plt.title("Scree plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.show()



##############################################

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=10, random_state = 0)
X_Reduite = svd.fit_transform(X)




dataFrame_Reduite=pd.DataFrame(X_Reduite) 
dataFrame_Reduite[dataFrame_Reduite.shape[1]]=Y



######################## SVM  #####################################

accur_lin = []
for i in range(0,10):
    print("Making sets %s" %i)
    dataFrame_Reduite = shuffle(dataFrame_Reduite)
    X=dataFrame_Reduite.iloc[:,0:-1].values
    Y=dataFrame_Reduite.iloc[:,-1].values
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10,random_state=2)# 80% training and 30% test
    # Random Forest model
    svclassifier.fit(X_train, y_train)
    y_preds = svclassifier.predict(X_test)
    accu_score=accuracy_score(y_test,y_preds)
    print("SVM accueacy : ",accu_score)
    accur_lin.append(accu_score) #Store accuracy in a list
print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs






######################### Split train+test #######################################
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,random_state=2)# 80% training and 30% test



######################## Decision Tree  #####################################
# Create Decision Tree classifer object
from sklearn.tree import DecisionTreeClassifier 
DTC = DecisionTreeClassifier()
# Train Decision Tree Classifer
DTC.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = DTC.predict(X_test) 
#Decision Tree Classifier accueacy
# Model Accuracy, how often is the classifier correct?
from sklearn.metrics import accuracy_score
print("DTC accueacy : ",accuracy_score(y_test,y_pred))


######################## Random Forest  #####################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
forest_model = RandomForestClassifier(random_state=2,n_estimators=100)
forest_model.fit(X_train, y_train)
melb_preds = forest_model.predict(X_test)
print("RFC accueacy : ",accuracy_score(y_test,melb_preds))


######################## SVM #####################################

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

#svclassifier = SVC(kernel='linear',C=100.0, random_state=42)
svclassifier.fit(X_train, y_train)
#Making Predictions
y_pred = svclassifier.predict(X_test) 
######################## Evaluating the Algorithm ########################
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
print("SVM accueacy : ",accuracy_score(y_test,y_pred))



#################################### Test #############################################
directory_Test='E:\\Master\\MS3_2019_2020\\Image mining\\miniProjet\\emotinrecognition\\yassin\\test'

#directory of PrivateTest
directory_Test='E:\Master\MS3_2019_2020\Image mining\miniProjet\emotinrecognition\yassin\test'
directory_Test='E:\\Master\\MS3_2019_2020\\Image mining\\miniProjet\\emotinrecognition\\coco\\PublicTest'
directory_Test='E:\\Master\\MS3_2019_2020\\Image mining\\miniProjet\\emotinrecognition\\alldatasetImages'
# emotions VECTOR 
emotions_Test = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] #Emotion list

# features extraction 
xx_test, yy_test = make_sets(directory_Test,emotions_Test)


X,Y=xx_test,yy_test

#Z=np.concatenate((X, xx_test), axis=0)
#P=np.concatenate((Y, yy_test), axis=0)



########################## Standarisation #######################################
from sklearn.preprocessing import StandardScaler as SS
ssX= SS()
X=ssX.fit_transform(X)


###################################################################################
######################### Split train+test #######################################
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,random_state=2)# 80% training and 30% test






melb_preds = forest_model.predict(X)
print("RFC accueacy : ",accuracy_score(yy_test,melb_preds))


























#Once our Linear SVM is trained, we can use it to classify subsequent texture images:
# loop over the testing images
for imagePath in paths.list_images(directory_PrivateTest):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = LBP_Features(gray)
    # model choice 
    prediction = DTC.predict(hist.reshape(1, -1))
    
    # display the image and the prediction
    cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

#    if cv2.waitKey(1) & 0xff == ord('q'):
#        break
cv2.destroyAllWindows()







############################# save object  ###################################
import pickle
#save object 'dictionary' in file dictionary.txt  
dictionary = {'DTC':DTC,'forest_model':forest_model,'dataFrame':dataFrame,'dataFrame_Reduite':dataFrame_Reduite} 
pickle.dump(dictionary, open('E:/Master/MS3_2019_2020/Image mining/miniProjet/emotinrecognition/dictionary.txt', 'wb'))
  
#read file dictionary.txt  
dictionary = pickle.load(open('E:/Master/MS3_2019_2020/Image mining/miniProjet/emotinrecognition/dictionary.txt', 'rb'))
# After dictionary is read from file
svclassifier=dictionary['svclassifier'] 
DTC=dictionary['DTC'] 
forest_model=dictionary['forest_model'] 
dataFrame=dictionary['dataFrame'] 
dataFrame_Reduite=dictionary['dataFrame_Reduite'] 


X=dataFrame.iloc[:,0:-1].values
Y=dataFrame.iloc[:,-1].values





import time
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


########################## Standarisation #######################################
from sklearn.preprocessing import StandardScaler as SS
ssX= SS()
X=ssX.fit_transform(X)


###################################################################################
######################### Split train+test #######################################
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,random_state=2)# 80% training and 30% test




n_estimators = 10
start = time.time()
svclassifier = OneVsRestClassifier(SVC(kernel='linear', probability=True), n_jobs=-1)

svclassifier.fit(X_train, y_train)
from sklearn.metrics import classification_report,accuracy_score
print("SVM accueacy : ",accuracy_score(y_test,y_pred))
end = time.time()
print ("Bagging SVC", end - start, svclassifier.score(y_test,y_pred))
proba = svclassifier.predict_proba(X)

