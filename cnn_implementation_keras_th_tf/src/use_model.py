import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.models import load_model
# Saving and loading model and weights
from keras.models import model_from_json

num_channel = 1

model =load_model("../Temp/model.model")

# load json and create model
json_file = open('../Temp/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model

loaded_model.load_weights("../Temp/model.h5")
print("Loaded model from disk")

model.save('../Temp/model.hdf5')
loaded_model=load_model('../Temp/model.hdf5')



# Testing a new image
test_image = cv2.imread('../Temp/horse-134.jpg')
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(200,200))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)

if num_channel==1:
    if K.image_dim_ordering()=='th':
        test_image= np.expand_dims(test_image, axis=0)
        test_image= np.expand_dims(test_image, axis=0)
        print (test_image.shape)
    else:
        test_image= np.expand_dims(test_image, axis=3) 
        test_image= np.expand_dims(test_image, axis=0)
        print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	
# Predicting the test image
print((model.predict(test_image)))
print(model.predict_classes(test_image))

