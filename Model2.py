
# coding: utf-8

# In[11]:


from __future__ import absolute_import
from __future__ import print_function
import os
import glob
import random
import numpy as np
from keras import optimizers
from keras.layers import LSTM
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.layers.wrappers import TimeDistributed
from keras.applications.mobilenet import MobileNet
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Input, InputLayer
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.applications import imagenet_utils


# In[3]:


videoFiles = glob.glob('../dataset/first-set/numpys/*.npy')
mosFiles = [i for i in videoFiles if 'mos' in i]
videoFiles = [i for i in videoFiles if 'mos' not in i]


# In[4]:


def myGenerator():
    while True:
        index_list = random.sample(range(1, 80), 3)
        alldata_x = []
        alldata_y = []
        for i in index_list:
            f = videoFiles[i]
            s = f[:-4]+'_mos.npy'
            a = np.load(f)
            b = np.load(s)
            alldata_x.append(a)
            alldata_y.append(b[0])
        alldata_x = np.array(alldata_x)
        #alldata_x = np.rollaxis(alldata_x, 1, 5)  
        #alldata_x = alldata_x.reshape((32, 30, height, width, 3))
        #alldata_x = np.swapaxes(alldata_x, 1, 4)
        alldata_y = np.array(alldata_y)
        yield alldata_x, alldata_y
#x = myGenerator()
#xtrain, ytrain = next(x)
#print('xtrain shape:',xtrain.shape)
#print('ytrain shape:',ytrain.shape)

# In[5]:


height = 68
width = 120
input_shape=(200, height, width, 3)


# In[12]:


def mySegNet(input_shape):
    base_model  = MobileNet(input_shape=(224,224,3), include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs=base_model.input, outputs=x)
    
    model = Sequential();
    #model.add(InputLayer(input_shape=input_shape))
    model.add(TimeDistributed(cnn_model, input_shape=input_shape))
    model.add(TimeDistributed(Flatten()))
    #model.add(cnn_model)
    #model.add(Flatten())
    
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())
    return model 
#mySegNet(input_shape)


# In[9]:


model = mySegNet(input_shape)

model.fit_generator(generator=myGenerator(),
                    use_multiprocessing=True,
                   steps_per_epoch=3, epochs=10)
model.save('model1.h5')
model.save_weights('model_weights1.h5')


# In[11]:


input_shape=(30, height, width, 3)
model = mySegNet(input_shape)
model.load_weights('model_weights2.h5')
totalTestSamples = len(allfiles)
predictions = []
ytrue = []
for i in range(0, totalTestSamples, batchSize):
    x = myTestDataGenerator()
    xtest, ytest = next(x)
    ytrue.append(ytest)
    pred = model.predict(xtest, batch_size=batchSize)
    for p in pred:
        predictions.append(p)
print('predictions shape: ', np.array(predictions).shape)


# In[60]:


tileFrames = []
for sample in ytrue[:1]:
    for frames in sample:
        t = []
        for frame in frames:
            f = []
            for i, j in enumerate(frame):
                if j!=0:
                    f.append(i+1)
            tileFrames.append(f)
print(np.array(tileFrames).shape)


# In[59]:


pTileFrames = []
for sample in predictions[:3]:
    for frames in sample:
        f = []
        for i, j in enumerate(frames):
            if j!=0:
                f.append(i+1)
        pTileFrames.append(f)
print(np.array(pTileFrames).shape)


# In[82]:


from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

breadth = 3840
width = 1920
tileSize = 192
tilesInColumn = width / tileSize
for i, tiles in enumerate(tileFrames):
    frame = np.zeros(width*breadth)
    print(tiles)
    for tileNo in tiles:
        tileRowNumber = int((tileNo - 1) / tilesInColumn)
        tileColumnNumber = (tileNo - 1) % tilesInColumn
        firstPixel = tileRowNumber * width * tileSize + tileColumnNumber * tileSize
        for rowPixel in range(0, tileSize):
            for columnPixel in range(0, tileSize):
                frame[int(firstPixel + rowPixel * breadth + columnPixel)] = 255
    frame = frame.reshape((width, breadth))
    plt.imshow(frame, interpolation='nearest')
    plt.show()
    break


# In[83]:


for i, tiles in enumerate(pTileFrames):
    frame = np.zeros(width*breadth)
    for tileNo in tiles:
        tileRowNumber = int((tileNo - 1) / tilesInColumn)
        tileColumnNumber = (tileNo - 1) % tilesInColumn
        firstPixel = tileRowNumber * width * tileSize + tileColumnNumber * tileSize
        for rowPixel in range(0, tileSize):
            for columnPixel in range(0, tileSize):
                frame[int(firstPixel + rowPixel * breadth + columnPixel)] = 255
    frame = frame.reshape((width, breadth))
    plt.imshow(frame, interpolation='nearest')
    plt.show()
    break


# In[ ]:


index  = 28
thresh = 0.5

temp = predictions[0][index] 
temp[temp > thresh] = 1
temp[temp <= thresh] = 0

for i, j in enumerate(ytest[0][index]):
    if ytest[0][index][i] != temp[i]:
        print('Index: ', i, 'Value: ', ytest[0][index][i], temp[i])


# In[ ]:


print(ytest[0][index].shape)

