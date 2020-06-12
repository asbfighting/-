from keras.preprocessing import image   #导入原始噪声图像
import matplotlib.pyplot as plt
import numpy as np
import random
jpg1="C:/Users/Think/Desktop/1.jpg"
Image=image.load_img(jpg1)
plt.imshow(Image)
plt.show()
Image.save("C:/Users/Think/Desktop/1.jpg")



jpg2=Image        #对图像添加随机白色像素点
jpg2=image.img_to_array(jpg2)

for i in range(3000):
    wn_X=random.randint(0,500)
    wn_Y=random.randint(0,500)
    jpg2[wn_X,wn_Y,:]=255
jpg2=image.array_to_img(jpg2)
plt.imshow(jpg2)
plt.show()
jpg2.save("C:/Users/Think/Desktop/2.jpg")




import numpy as np             #导入添加白像素点后的网络图像
import cv2
import matplotlib.pyplot as plt
img=cv2.imread("C:/Users/Think/Desktop/2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

img=np.uint8(img)        3使用opencv的fastNIMeansDenoisingColored内置噪声过滤算法过滤图像噪声
jpg2= cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
plt.rcParams['figure.figsize']=(15.0,15.0)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(jpg2)
plt.show()
