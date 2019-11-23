#!/usr/bin/env python
# coding: utf-8

# # image processing

# In[206]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys


# In[207]:


print(os.getcwd())
DATADIR=os.getcwd()

# In[276]:


CATEGORIES = ["Dog", "Cat"]

training_data = []
IMG_SIZE = 50

for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        
        for img in os.listdir(path):  # iterate over each image per dogs and cats
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)  # convert to array
                img_array = cv2.resize(img_array, (200, 200))
                #plt.imshow(img_array,cmap=plt.cm.binary)
                plt.imshow(img_array,cmap='gray')
                plt.show()
                break
        break
                


# In[277]:


print(img_array.shape)


# In[278]:


#Loading in your own data in differnt floders

DATADIR = os.getcwd()

CATEGORIES = ["Dog", "Cat"]

training_data = []
IMG_SIZE = 100

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        print("folder path --->>",path,class_num)

        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)  # convert to array
                #img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
                print('image path ->',os.path.join(path,img))

                print('img_array',img_array.shape)
                newimg_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                print('newimg_array',newimg_array.shape)
                training_data.append([newimg_array, class_num])  # add this to our training_data
                
                #cv2.imshow('img',newimg_array)
                #cv2.waitKey(50)
                #cv2.destroyAllWindows()
            except Exception as e:  # in the interest in keeping the output clean...
                print("general exception", e, os.path.join(path,img))
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path,img))
                


# In[279]:


create_training_data()


# In[281]:


print('training_data:',len(training_data))
#print(training_data)


# In[282]:


import random
random.shuffle(training_data)


# In[284]:


# we've got the classes nicely mixed in! Time to make our model!
for sample in training_data[:10]:
    print(sample[1])


# In[285]:


X = []
y = []

for features,label in training_data:
    X.append((features))
    y.append(label)


# In[286]:


X =np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)#(-1,row,col,chanale)
X= X.astype('float32')


# In[287]:


import pickle
pic_out =open('X.pickle','wb')
pickle.dump(X,pic_out)
pic_out.close()

pic_out =open('y.pickle','wb')
pickle.dump(y,pic_out)
pic_out.close()


# In[288]:


#---------------model ----------------------------


# In[289]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# In[290]:


import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)


# In[291]:


#print(X[0])


# In[292]:


#normalize
#X = tf.keras.utils.normalize(X, axis=1)
#y = tf.keras.utils.normalize(y, axis=1)
#----------------or -------------------
X = X/255.0
#print(X[0])
#plt.imshow(X[0:],cmap='gray')
#plt.show()


# In[293]:


model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3)


# In[294]:


x_test = X
y_test =y


# In[295]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)


# In[296]:


#saveing get mode by using keras like pickel
#save
model.save('cat_dog_model.model')
#get
cat_dog_model = tf.keras.models.load_model('cat_dog_model.model')


# In[297]:

model.summary()

#predictions
predictions = cat_dog_model.predict(X)
print(predictions)


# In[298]:


import numpy as np
print(np.argmax(predictions[0]))


# In[299]:


#plt.imshow(X[0],cmap='gray')
#plt.show()


# In[300]:


a=[]
for i in range(200):
    b=np.argmax(predictions[i])
    a.append(b)


# In[301]:


print(a)


# In[302]:


a.count(0)


# In[ ]:




