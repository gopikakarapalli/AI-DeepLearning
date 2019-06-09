# Import libraries
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.utils import np_utils
from keras.models import Sequential

########################################################################
from keras import backend as K
K.set_image_dim_ordering('tf') #for tensorflow img.shape =r,col,ch
#K.set_image_dim_ordering('th') # for theano  img.shape =ch,r,col
#--------------or--------------------
#change json file tensorflow <==> theano
# for tensorflow img.shape =r,col,ch
# for theano  img.shape =ch,r,col
############################################################################



PATH = os.getcwd()
DataPath = os.path.join( PATH,'data')
DataDirList = os.listdir(DataPath)
print(DataDirList)

dataset=[]
resize = 200
backend ='tf'

num_channel = 1

def create_training_data():
	for EachDir in DataDirList:
		DirPath = os.path.join(DataPath,EachDir)
		classNum = DataDirList.index(EachDir)
		print(DirPath,'----------------->',classNum)
		
		for img in os.listdir(DirPath):
			input_img=cv2.imread(os.path.join(DirPath,img),cv2.IMREAD_GRAYSCALE)
			input_img=cv2.resize(input_img,(resize,resize))
			dataset.append(input_img)

			print('image path ->',os.path.join(DirPath,img))
			print('input_img shape',input_img.shape )
			#plt.imshow(input_img,cmap='gray')
			#plt.show()

create_training_data()

#normalize
dataset = np.array(dataset)
dataset = dataset.astype('float32')
dataset /= 255
print (dataset.shape)

# -----or ------
#X = keras.utils.normalize(X, axis=1)


##################################################################################
if num_channel==1:
	if K.image_dim_ordering()==backend:              
		dataset= np.expand_dims(dataset, axis=1)  #(no.img,row,col) ==> (no.img,ch,row,col)
		print (dataset.shape)
	else:
		dataset= np.expand_dims(dataset, axis=4)  # (no.img,row,col) ==> #(no.img,row,col.ch)
		print (dataset.shape)                     			
else:
	if K.image_dim_ordering()==backend:
		dataset=np.rollaxis(dataset,3,1)       # (no.img,row,col,ch) ==>(no.img,ch,row,col)
		print (dataset.shape)
'''
#*************************************  USE_SKLEARN FOR PREPROCESSING ****************************
USE_SKLEARN_PREPROCESSING=False

if USE_SKLEARN_PREPROCESSING:
	from sklearn import preprocessing
	
	def image_to_feature_vector(image, size=(resize, resize)):
		# resize the image to a fixed size, then flatten the image into
		# a list of raw pixel intensities
		return cv2.resize(image, size).flatten()
	
	dataset=[]
	for EachDir in DataDirList:
		DirPath = os.path.join(DataPath,EachDir)
		classNum = DataDirList.index(EachDir)
		print(DirPath,'----------------->',classNum)

		for img in os.listdir(DirPath):
			input_img=cv2.imread(os.path.join(DirPath,img),cv2.IMREAD_GRAYSCALE)
			input_img_flatten=image_to_feature_vector(input_img,(resize,resize))
			print('input_img_flatten shape',input_img_flatten.shape )
			dataset.append(input_img_flatten)
			print('image path ->',os.path.join(DirPath,img))
			print('input_img shape',input_img.shape )
			#plt.imshow(input_img,cmap='gray')
			#plt.show()
	
	dataset = np.array(dataset)
	dataset = dataset.astype('float32')
	print (dataset.shape)
	dataset_scaled = preprocessing.scale(dataset)
	print ('dataset_scaled',dataset_scaled.shape)
	
	print (dataset_scaled.mean(axis=0))
	print (dataset_scaled.std(axis=0))

	if K.image_dim_ordering()=='tf':
		dataset_scaled=dataset_scaled.reshape(dataset.shape[0],num_channel,resize,resize)
		print (dataset_scaled.shape)
		
	else:
		dataset_scaled=dataset_scaled.reshape(dataset.shape[0],resize,resize,num_channel)
		print (dataset_scaled.shape)
	
	
	if K.image_dim_ordering()=='tf':
		dataset_scaled=dataset_scaled.reshape(dataset.shape[0],num_channel,resize,resize)
		print (dataset_scaled.shape)
		
	else:
		dataset_scaled=dataset_scaled.reshape(img_data.shape[0],resize,resize,num_channel)
		print (dataset_scaled.shape)

if USE_SKLEARN_PREPROCESSING:
	dataset=dataset_scaled

#************************** END USE_SKLEARN FOR PREPROCESSING ************************************************
'''

# Define the number of classes
num_classes = 4

num_of_samples = dataset.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:202]=0
labels[202:404]=1
labels[404:606]=2
labels[606:]=3
	  
names = ['cats','dogs','horses','humans']
	  
# convert class labels to on-hot encoding
labels = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
data = shuffle(dataset,labels, random_state=2)

import pickle

pickle.dump(data,open('image preprocessing.pickle','wb'))
print('data file saved')








