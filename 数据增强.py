from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
Data_boost=ImageDataGenerator(rotation_range=60,height_shift_range=0.3,width_shift_range=0.2,zoom_range=0.2,shear_range=0.3)
cat_path='F:/jupyter notebook/Deep leariing——图像可视化/cat1.jpg'    #替换为自己实际的原始图片路径
Image=image.load_img(cat_path,target_size=(150,150))
Image=image.img_to_array(Image)
Image=Image.reshape(1,150,150,3)

i=0
for x in Data_boost.flow(Image,batch_size=1):
       plt.figure(i)
       imgplot= plt.imshow(image.array_to_img(x[0]))
       i+=1
       if i%4==0:
         break
