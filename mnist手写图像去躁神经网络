from __future__ import print_function 
import os
import struct
import numpy as np 
import keras
from keras.datasets import mnist
from keras import models,layers
from keras import backend as K 
from keras.preprocessing import image
import cv2

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8)) 
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        return images, labels

    
X_train, y_train = load_mnist('F:/jupyter notebook/Deep leariing——图像可视化/data',kind='train')   #替换为自己的mnist数据解压包地址
X_test, y_test = load_mnist('F:/jupyter notebook/Deep leariing——图像可视化/data',kind='t10k')
x_train = X_train.astype('float32') / 255. 
x_test = X_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28,1))
x_test = np.reshape(x_test, (len(x_test), 28, 28,1))
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.) 
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

batch_size = 128 
num_classes = 10
epochs = 1
img_rows, img_cols = 28, 28

model=models.Sequential()
model.add(layers.Conv2D(32,(4,4),activation='relu',padding='same',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2),padding='same'))
model.add(layers.Conv2D(32,(4,4),activation='relu',padding='same'))
model.add(layers.MaxPooling2D((2,2),padding='same'))
model.add(layers.Conv2D(16,(4,4),activation='relu',padding='same'))
model.add(layers.UpSampling2D((2,2)))  
model.add(layers.Conv2D(16,(4,4),activation='relu',padding='same'))
model.add(layers.UpSampling2D((2,2)))    
model.add(layers.Conv2D(1,(4,4),activation='sigmoid',padding='same'))   
model.summary()

model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.fit(x_train_noisy, x_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test_noisy, x_test))
score = model.evaluate(x_test_noisy, x_test, verbose=0)



for currentEpoch in range(1,3):
    ii=currentEpoch+1 
    dir = 'D:/temp/'+str(ii)
    if not os.path.isdir(dir) :
        print("mkdir:"+dir)
        os.mkdir(dir)
        model.fit(x_train_noisy, x_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test_noisy, x_test))
        index = 0 
        image1=x_test_noisy[index] 
        image2=x_test[index]
        result = model.predict(image1.reshape(1,28,28,1))
        result = result.reshape(28,28,1) 
        result *= 255
        cv2.imwrite(dir+'/result.jpg',result)
        image1 = image1.astype('unit8') 
        image2 = image2.astype('unit8')     
        image1 *= 255
        image2 *= 255
        image1=image1.reshape(28,28,1) 
        image2=image2.reshape(28,28,1)
        noise = X_test[index] + 0.5 * np.random.normal(loc=0.0, scale=100.0, size=X_train[index].shape) 
        noise = noise.reshape(28,28,1) 
        cv2.imwrite(dir+'/noise.jpg',noise)
        cv2.imwrite(dir+'/image2.jpg',image2)
