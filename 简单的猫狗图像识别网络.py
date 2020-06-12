#本文件为使用dropout数据削弱与keras.preprocessing包中图像预处理增强模块后的对比， 处理后明显从验证准确率、曲线贴合程度均有所提高
#文中所有的路径根据自己猫狗数据集对应的存放地址适当修改
#coding=utf8
import os ,shutil
import PIL
original_dataset_dir='C:/Users/Think/Desktop/dog and cats/train/train'   #下载数据的初始路径及保留地址
base_dir='C:/Users/Think/Desktop/dog and cats/zzh2'      #在初始路径下新建一个根目录，用于后续验证、测试等集划分
os.mkdir(base_dir)


train_data='C:/Users/Think/Desktop/dog and cats/zzh2/train_data'
os.mkdir(train_data)
test_data='C:/Users/Think/Desktop/dog and cats/zzh2/test_data'
os.mkdir(test_data)
valid_data=valid_data='C:/Users/Think/Desktop/dog and cats/zzh2/valid_data'     #os.mkdir其作用是在指定路径下生成要求的文件，不具备内容植入及拷贝作用
os.mkdir(valid_data)

train_data='C:/Users/Think/Desktop/dog and cats/zzh2/train_data'
os.path.join(base_dir,train_data)
test_data='C:/Users/Think/Desktop/dog and cats/zzh2/test_data'        #os.path.join用于将内容拷贝入指定文件夹，而不具备创建文件夹作用
os.path.join(base_dir,test_data)
valid_data=valid_data='C:/Users/Think/Desktop/dog and cats/zzh2/valid_data'
os.path.join(base_dir,valid_data)

cats_train='C:/Users/Think/Desktop/dog and cats/zzh2/train_data/cats'
os.mkdir(cats_train)
dogs_train='C:/Users/Think/Desktop/dog and cats/zzh2/train_data/dogs'
os.mkdir(dogs_train)                                                     #创建猫和狗的对应目录 

cats_test='C:/Users/Think/Desktop/dog and cats/zzh2/test_data/cats'
os.mkdir(cats_test)
dogs_test='C:/Users/Think/Desktop/dog and cats/zzh2/test_data/dogs'     #创建猫和狗的测试集目录
os.mkdir(dogs_test)

cats_valid='C:/Users/Think/Desktop/dog and cats/zzh2/valid_data/cats'
os.mkdir(cats_valid)
dogs_valid='C:/Users/Think/Desktop/dog and cats/zzh2/valid_data/dogs'   #创建猫和狗的验证集目录
os.mkdir(dogs_valid) 

dataset_overall_one=['cat.{}.jpg'.format(i) for i in range(500) ]#使用shutil包向创建的文件夹内分别导入对应的数据,使用format函数格式化字符串，类似C语言中printf的%d与后对应输出关系
for i in dataset_overall_one:
 src=os.path.join(original_dataset_dir,i)
 dst=os.path.join(cats_valid,i)
 shutil.copyfile(src,dst)

dataset_overall_two=['dog.{}.jpg'.format(i) for i in range(500,1000) ] 
for i in dataset_overall_two:
 src=os.path.join(original_dataset_dir,i)
 dst=os.path.join(dogs_valid,i)
 shutil.copyfile(src,dst)



dataset_overall_three=['cat.{}.jpg'.format(i) for i in range(1000,2000) ] 
for i in dataset_overall_three:
 src=os.path.join(original_dataset_dir,i)
 dst=os.path.join(cats_train,i)
 shutil.copyfile(src,dst)

dataset_overall_four=['dog.{}.jpg'.format(i) for i in range(2000,3000) ] 
for i in dataset_overall_four:
 src=os.path.join(original_dataset_dir,i)
 dst=os.path.join(dogs_train,i)
 shutil.copyfile(src,dst)


dataset_overall_five=['cat.{}.jpg'.format(i) for i in range(3000,3500) ] 
for i in dataset_overall_five:
 src=os.path.join(original_dataset_dir,i)
 dst=os.path.join(cats_test,i)
 shutil.copyfile(src,dst)


dataset_overall_six=['dog.{}.jpg'.format(i) for i in range(3500,4000) ] 
for i in dataset_overall_six:
 src=os.path.join(original_dataset_dir,i)   #使用shutil包成功复制划分并导入了训练、测试、验证图像
 dst=os.path.join(dogs_test,i) 
 shutil.copyfile(src,dst)
 
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
test_datagen=ImageDataGenerator(rescale=1./255)   #数据增强适用于样本数量少时的防止过拟合，不能对验证数据使用

train_generator = train_datagen.flow_from_directory(train_data,target_size=(150, 150),batch_size=32,class_mode='binary')
valid_generator = test_datagen.flow_from_directory(valid_data,target_size=(150, 150),batch_size=32,class_mode='binary')

from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))  #Conv1D序列的输入形状为两个维度，单纯数据序列为一个维度，可使用reshape解决；Conv2D输入为3个维度，常用语图像处理等方面
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())                                #使用Flatten将其转化为Dense层所需的一维输入层
model.add(layers.Dropout(0.5))                             #使用Dropout防止小数据量的过拟合
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))            #搭建图像处理的Conv2D框架
 
model.summary()

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])  #设定SGD梯度下降学习速率调整的相关参数
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=50,validation_data=valid_generator,validation_steps=50)#使用生成器模拟输入与训练数据

import matplotlib.pyplot as plt   #绘制训练过程中输出损失和精度的变化趋势

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(accuracy) + 1)

plt.rcParams['figure.figsize']=(20.0,10.0)

plt.subplot(121)
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(122)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

