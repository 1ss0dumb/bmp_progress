import cv2 as cv
# from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog,QDialog,QGraphicsView

# from PyQt5 import QtCore, QtGui, QtWidgets
import math
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QDialog, QFileDialog, QApplication, QGraphicsPixmapItem,QGraphicsScene
import bmp_1
import sys
import random
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# from  win_2 import *
from mainform1 import *

The_file=''
img_np=[]
img_height=0
img_width=0
bit_num=0
img_path=''
def clamp(pv):
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv
        
class MyFigure(FigureCanvas):
    def __init__(self,width=5, height=4, dpi=100):
        #第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        #第二步：在父类中激活Figure窗口
        super(MyFigure,self).__init__(self.fig) #此句必不可少，否则不能显示图形
        #第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111)
    #第四步：就是画图，【可以在此类中画，也可以在其它类中画】
    # def plotsin(self):
    #     self.axes0 = self.fig.add_subplot(111)
    #     t = np.arange(0.0, 3.0, 0.01)
    #     s = np.sin(2 * np.pi * t)
    #     self.axes0.plot(t, s)
    # def plotcos(self):
    #     t = np.arange(0.0, 3.0, 0.01)
    #     s = np.sin(2 * np.pi * t)
    #     self.axes.plot(t, s)

class My_UI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle('窗口标题')

    def open_see_bmp_pic(self):
        '''
        从本地读取图片
        '''
        # 打开文件选取对话框
        filename, _ = QFileDialog.getOpenFileName(self, '打开图片')
        if filename:
            global The_file,img_height, img_width, bit_num,img_np,img_path
            The_file=filename
            img_np, img_height, img_width, bit_num, img_path = bmp_1.bmp_img_read_hist(filename)
            img_np=np.array(img_np,dtype='uint8')
            x = img_np.shape[1]                                 #获取图像大小
            y = img_np.shape[0]
            self.zoomscale=1                               #图片放缩尺度
            frame = QImage(img_np, x, y ,x * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
            self.item.setScale(self.zoomscale)
            self.scene=QGraphicsScene()                             #创建场景
            self.scene.addItem(self.item)
            self.graphicsView.setScene(self.scene) 


    def open_see_gray_bmp(self):
        '''
        从本地读取图片
        '''
        global The_file,img_height, img_width, bit_num,img_np,img_path
        if The_file=='':
            The_file, _ = QFileDialog.getOpenFileName(self, '打开图片')
        filename=The_file
        img_np, img_height, img_width, bit_num, img_path = bmp_1.bmp_img_read_gray(filename)
        img_np=np.array(img_np,dtype='uint8')
        x = img_np.shape[1]                                 #获取图像大小
        y = img_np.shape[0]
        self.zoomscale=1                               #图片放缩尺度
        frame = QImage(img_np, x, y ,x * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        
            
    def the_info_show(self):
        global img_height, img_width, bit_num ,img_np,img_path
        self.lineEdit.setText(str(img_width))
        self.lineEdit_2.setText(str(img_height))
        self.lineEdit_3.setText(str(bit_num))
        self.lineEdit_5.setText(str(img_width))
        self.lineEdit_6.setText(str(img_height))
        tem_path=os.path.split(img_path)[1]
        self.lineEdit_4.setText(tem_path)
        self.F = MyFigure(width=4, height=3, dpi=100)
        # arr = img_np.flatten()
        ar = np.array(img_np[:,:,0]).flatten()
        self.F.axes.hist(ar, bins=2**bit_num,facecolor='r', alpha=0.75)
        ag = np.array(img_np[:,:,1]).flatten()
        self.F.axes.hist(ag, bins=2**bit_num,facecolor='g', alpha=0.75)
        ab = np.array(img_np[:,:,2]).flatten()
        self.F.axes.hist(ab, bins=2**bit_num,facecolor='b', alpha=0.75)
        self.scene1 = QGraphicsScene()  #创建一个场景
        self.scene1.addWidget(self.F)   #将图形元素添加到场景中
        self.graphicsView_2.setScene(self.scene1) #将创建添加到图形视图显示窗口
    
    def equal_zoom_in_click(self):
        self.zoomscale=self.zoomscale-0.05
        if self.zoomscale<=0:
            self.zoomscale=0.2                         #图片放缩尺度
        self.item.setScale(self.zoomscale)
    
    def equal_zoom_out_click(self):
        self.zoomscale=self.zoomscale+0.05
        if self.zoomscale>=1.2:
            self.zoomscale=1.2
        self.item.setScale(self.zoomscale)   
    
    def roll_the_img(self):
        global img_np
        y, x, channels = img_np.shape
        rotate = cv.getRotationMatrix2D((x*0.5, y*0.5), 90, 1)
        res = cv.warpAffine(img_np, rotate, (x, y))
        frame = QImage(res, x, y , x*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        img_np=res
        
    def roll_the_img_anti(self):
        global img_np
        y, x, channels = img_np.shape
        rotate = cv.getRotationMatrix2D((x*0.5, y*0.5), 270, 1)
        res = cv.warpAffine(img_np, rotate, (x, y))
        frame = QImage(res, x, y , x*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        img_np=res
        
    # def roll_the_img_all(self):
    #     global img_np
    #     arc = int(self.lineEdit_7.text())
    #     y, x, channels = img_np.shape
    #     print(x,y)
    #     angel=(arc/180)*math.pi
    #     cosa=abs(math.cos(angel))
    #     sina=abs(math.sin(angel))
    #     rotate = cv.getRotationMatrix2D((x*0.5, y*0.5), arc, 1)
    #     nx=int(x*cosa+y*sina)
    #     ny=int(x*sina+y*cosa)
    #     print(nx,ny)
    #     res = cv.warpAffine(img_np, rotate, (nx,ny))
    #     frame = QImage(res, nx, ny , nx*3 ,QImage.Format_RGB888)
    #     pix = QPixmap.fromImage(frame)
    #     self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
    #     # self.item.setScale(self.zoomscale)
    #     self.scene=QGraphicsScene()                             #创建场景
    #     self.scene.addItem(self.item)
    #     self.graphicsView.setScene(self.scene)
    #     img_np=res
           
    

    def roll_the_img_all(self):
        global img_np
        if self.lineEdit_7.text()!='':
            angle = int(self.lineEdit_7.text())
        else:
            angle = 0
        height, width = img_np.shape[:2]
        if int(angle / 90) % 2 == 0:
            reshape_angle = angle % 90
        else:
            reshape_angle = 90 - (angle % 90)
        reshape_radian = math.radians(reshape_angle)  # 角度转弧度
        # 三角函数计算出来的结果会有小数，所以做了向上取整的操作。
        new_height = math.ceil(height * np.cos(reshape_radian) + width * np.sin(reshape_radian))
        new_width = math.ceil(width * np.cos(reshape_radian) + height * np.sin(reshape_radian))
        new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        
        radian = math.radians(angle)
        cos_radian = np.cos(radian)
        sin_radian = np.sin(radian)
        channel=3
        if channel:
            fill_height = np.zeros((height, 2, channel), dtype=np.uint8)
            fill_width = np.zeros((2, width + 2, channel), dtype=np.uint8)
        else:
            fill_height = np.zeros((height, 2), dtype=np.uint8)
            fill_width = np.zeros((2, width + 2), dtype=np.uint8)
        img_copy = img_np.copy()
        # 因为双线性插值需要得到x+1，y+1位置的像素，映射的结果如果在最边缘的话会发生溢出，所以给图像的右边和下面再填充像素。
        img_copy = np.concatenate((img_copy, fill_height), axis=1)
        img_copy = np.concatenate((img_copy, fill_width), axis=0)
        dx_back = 0.5 * width - 0.5 * new_width * cos_radian - 0.5 * new_height * sin_radian
        dy_back = 0.5 * height + 0.5 * new_width * sin_radian - 0.5 * new_height * cos_radian
        for y in range(new_height):
            for x in range(new_width):
                x0 = x * cos_radian + y * sin_radian + dx_back
                y0 = y * cos_radian - x * sin_radian + dy_back
                x_low, y_low = int(x0), int(y0)
                x_up, y_up = x_low + 1, y_low + 1
                u, v = math.modf(x0)[0], math.modf(y0)[0]  # 求x0和y0的小数部分
                x1, y1 = x_low, y_low
                x2, y2 = x_up, y_low
                x3, y3 = x_low, y_up
                x4, y4 = x_up, y_up
                if 0 < int(x0) <= width and 0 < int(y0) <= height:
                    pixel = (1 - u) * (1 - v) * img_copy[y1, x1] + (1 - u) * v * img_copy[y2, x2] + u * (1 - v) * img_copy[y3, x3] + u * v * img_copy[y4, x4]  # 双线性插值法，求像素值。
                    new_img[int(y), int(x)] = pixel
        ny,nx=new_img.shape[:2]
        frame = QImage(new_img, nx, ny , nx*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        img_np=new_img
        
    
    def resize_the_img(self):
        global img_np
        y, x, channels = img_np.shape
        n_x= int(self.lineEdit_5.text())
        n_y= int(self.lineEdit_6.text())
        res = cv.resize(img_np, (n_x, n_y))
        frame = QImage(res, n_x, n_y , n_x*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        img_np=res
    
    def pixel_probability(img):
        """
        计算像素值出现概率
        :param img:
        :return:
        """
        assert isinstance(img, np.ndarray)
    
        prob = np.zeros(shape=(256))
    
        for r2 in img:
            for c2 in r2:
                prob[c2] += 1
    
        r, c = img.shape
        prob = prob / (r * c)
    
        return prob
    

    def probability_to_histogram(self):
        """
        根据像素概率将原始图像直方图均衡化
        :param img:
        :param prob:
        :return: 直方图均衡化后的图像
        """
        global img_np
        assert isinstance(img_np, np.ndarray)

        (b, g, r) = cv.split(img_np)
        equal_b = cv.equalizeHist(b)
        equal_g = cv.equalizeHist(g)
        equal_r = cv.equalizeHist(r)
        dst = cv.merge((equal_b, equal_g, equal_r))

        img_np=dst
        y, x, channels = img_np.shape
        self.F = MyFigure(width=4, height=3, dpi=100)
        ar = np.array(img_np[:,:,0]).flatten()
        self.F.axes.hist(ar, bins=2**bit_num,facecolor='r', alpha=0.75)
        ag = np.array(img_np[:,:,1]).flatten()
        self.F.axes.hist(ag, bins=2**bit_num,facecolor='g', alpha=0.75)
        ab = np.array(img_np[:,:,2]).flatten()
        self.F.axes.hist(ab, bins=2**bit_num,facecolor='b', alpha=0.75)
        self.scene1 = QGraphicsScene()  #创建一个场景
        self.scene1.addWidget(self.F)   #将图形元素添加到场景中
        self.graphicsView_2.setScene(self.scene1) #将创建添加到图形视图显示窗口
        
        frame = QImage(img_np, x, y , x*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
    

    def method_gdh(self):
        filename, _ = QFileDialog.getOpenFileName(self, '打开图片')
        if filename:
            global The_file,img_height, img_width, bit_num,img_np
            The_file=filename
            dst, new_height, new_width, new_num, new_path = bmp_1.bmp_img_read_hist(filename)
            dst=np.array(dst,dtype='uint8')
            y, x, channels = dst.shape
            frame = QImage(dst, x, y , x*3 ,QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
            # self.item.setScale(self.zoomscale)
            self.scene=QGraphicsScene()                             #创建场景
            self.scene.addItem(self.item)
            self.graphicsView_2.setScene(self.scene)
            color = ('b', 'g', 'r')
            def_img=dst
            for i, col in enumerate(color):
                hist1, bins = np.histogram(img_np[:, :, i].ravel(), 256, [0, 256])
                hist2, bins = np.histogram(dst[:, :, i].ravel(), 256, [0, 256])
                # 获得累计直方图
                cdf1 = hist1.cumsum()
                cdf2 = hist2.cumsum()
                # 归一化处理
                cdf1_hist = hist1.cumsum() / cdf1.max()
                cdf2_hist = hist2.cumsum() / cdf2.max()
        
                # diff_cdf 里是每2个灰度值比率间的差值
                diff_cdf = np.zeros((256,256))
                for j in range(256):
                    for k in range(256):
                        diff_cdf[j][k] = abs(cdf1_hist[j] - cdf2_hist[k])
                # FigA 中的灰度级与目标灰度级的对应表
                lut = np.zeros((256,2), dtype=np.int)
                for j in range(256):
                    squ_min = diff_cdf[j][0]
                    index = 0
                    for k in range(256):
                        if squ_min > diff_cdf[j][k]:
                            squ_min = diff_cdf[j][k]
                            index = k
                    lut[j] = ([j, index])
        
                h = int(img_np.shape[0])
                w = int(img_np.shape[1])
                
                # 对原图像进行灰度值的映射
                for j in range(h):
                    for k in range(w):
                        def_img[j, k, i] = lut[img_np[j, k, i]][1]
                
            y, x, channels = def_img.shape
            frame = QImage(def_img, x, y , x*3 ,QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
            # self.item.setScale(self.zoomscale)
            self.scene=QGraphicsScene()                             #创建场景
            self.scene.addItem(self.item)
            self.graphicsView.setScene(self.scene)
    
    def salt_pepper(self):
        global img_np
        y, x, channels = img_np.shape
        output = np.zeros(img_np.shape, np.uint8)
        prob=0.01
        thres = 1 - prob
        for i in range(img_np.shape[0]):
            for j in range(img_np.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img_np[i][j]
        frame = QImage(output, x, y , x*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        img_np=output
    
    
    def gause_noise(self):
        mean=0
        var=20
        global img_np
        img=np.copy(img_np)
        h, w, c = img.shape
        for row in range(h):
            for col in range(w):
                # 获取三个高斯随机数
                # 第一个参数：概率分布的均值，对应着整个分布的中心
                # 第二个参数：概率分布的标准差，对应于分布的宽度
                # 第三个参数：生成高斯随机数数量
                s = np.random.normal(loc=mean, scale=var, size=3)
                # 获取每个像素点的bgr值
                b = img[row, col, 0]  # blue
                g = img[row, col, 1]  # green
                r = img[row, col, 2]  # red
                # 给每个像素值设置新的bgr值
                img[row, col, 0] = clamp(b + s[0])
                img[row, col, 1] = clamp(g + s[1])
                img[row, col, 2] = clamp(r + s[2])
            # if row % 10 == 0:
            #     print("{:%}".format(row / h))
        frame = QImage(img, w, h , w*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        img_np=img
        
    def method_noise(self):
        global img_np
        dst=img_np
        y, x, channels = dst.shape
        for i in range(1000):
            m = np.random.randint(0, y)
            n = np.random.randint(0, x)
            dst[m, n, :] = 255
        frame = QImage(dst, x, y , x*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        img_np=dst

#高斯滤波
    def gose_polish(self):
        # 读取图片
        global img_np
        y, x, channels = img_np.shape
        # source = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
        kel=3
        # 方框滤波
        tem=self.lineEdit_9.text()
        if(tem!=''):
            if(int(tem)!=3):
                kel=int(tem)
        else:
            kel=3
        result = cv.GaussianBlur(img_np, (kel, kel), 0)
        
        # # 显示图形
        # titles = ['Source Image', 'GaussianBlur Image']
        # images = [source, result]
        
        # for i in range(2):
        #     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        #     plt.title(titles[i])
        #     plt.xticks([]), plt.yticks([])
        
        # plt.show()
        
        frame = QImage(result, x, y , x*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        img_np=result
        
        
# 均值滤波
    def mean_filter(self):
        global img_np
        y, x, channels = img_np.shape
        kel=3
        tem=self.lineEdit_10.text()
        if(tem!=''):
            if(int(tem)!=3):
                kel=int(tem)
        else:
            kel=3
        result = cv.blur(img_np, (kel,kel))  # 可以更改核的大小
        frame = QImage(result, x, y , x*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        img_np=result
    
# 中值滤波    
    def median_filter(self):
        global img_np
        y, x, channels = img_np.shape
        kel=3
        tem=self.lineEdit_8.text()
        if(tem!=''):
            if(int(tem)!=3):
                kel=int(tem)
        else:
            kel=3
        result = cv.medianBlur(img_np, kel)  # 可以更改核的大小
        frame = QImage(result, x, y , x*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        img_np=result       

   
    def SobelSharp1(self):
        global img_np
        h,w,channels=img_np.shape
        SobelX = [1,0,-1,2,0,-2,1,0,-1]
        SobelY = [-1,-2,-1,0,0,0,1,2,1]
        iSuanSharp = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)
        tmpX = [0]*9
        tmpY = [0]*9
        for i in range(1,h-1):
            for j in range(1,w-1):
                for k in range(3):
                    for l in range(3):
                        tmpX[k*3+l] = img_np[i-1+k,j-1+l,0]*SobelX[k*3+l]
                        tmpX[k*3+l] = img_np[i-1+k,j-1+l,0]*SobelY[k*3+l]
                tem=sum(tmpX)+sum(tmpY)
                iSuanSharp[i,j] = tem
        # print(iSuanSharp.shape)
        cv.imshow('iMaxSharp',iSuanSharp)
        plt.imshow(iSuanSharp.astype('uint8'))
        y, x, channels = iSuanSharp.shape
        frame = QImage(iSuanSharp, y, x ,QImage.Format_Indexed8)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        self.graphicsView_2.setScene(self.scene)
        
    def SobelSharp(self):
        global img_np
        img_gray = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)
        
        x = cv.Sobel(img_gray, cv.CV_16S, 1, 0)
        y = cv.Sobel(img_gray, cv.CV_16S, 0, 1)
        # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
        # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
        scale_abs_x = cv.convertScaleAbs(x)
        scale_abs_y = cv.convertScaleAbs(y)
        result = cv.addWeighted(scale_abs_x, 0.5, scale_abs_y, 0.5, 0)
        cv.imshow('Sobel',result)
        
    def Cannysharp(self):
        global img_np
        img_gray = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(img_gray, (3, 3), 0)  # 用高斯滤波处理原图像降噪
        canny = cv.Canny(image=blur, threshold1=50, threshold2=150, L2gradient=True)  # 50是最小阈值,150是最大阈值
        cv.imshow('Sobel',canny)
        
    def RobertSharp(self):
        global img_np
        result = np.copy(img_np)
        h, w, _ = result.shape
        rob = [[-1, -1], [1, 1]]
        for x in range(h):
            for y in range(w):
                if (y + 2 <= w) and (x + 2 <= h):
                    img_child = result[x:x + 2, y:y + 2, 1]
                    list_robert = rob * img_child
                    result[x, y] = abs(list_robert.sum())  # 求和加绝对值
        cv.imshow('Robert',result)

    #全局阈值        
    def global_th(self):
        global img_np
        img_gray = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)
        ret1, th1 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
        y, x = th1.shape
        frame = QImage(th1, x, y  ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        cv.imshow('global',th1)
    
    #ots阈值
    def ots_th(self):
        global img_np
        img_gray = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)
        ret1, th1 = cv.threshold(img_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        y, x = th1.shape
        frame = QImage(th1, x, y , x*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        cv.imshow('ots',th1)
        
    #自适应阈值
    def adp_th(self):
        global img_np
        img_gray = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)
        th1 = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        y, x = th1.shape
        frame = QImage(th1, x, y , x*3 ,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                             #创建场景
        self.scene.addItem(self.item)
        cv.imshow('adp',th1)

    
    
# image = cv.LoadImage('lena.jpg',0)

# PrewittX = [1,0,-1,1,0,-1,1,0,-1]
# PrewittY = [-1,-1,-1,0,0,0,1,1,1]
# IsotropicX = [1,0,-1,1.414,0,-1.414,1,0,-1]
# IsotropicY = [-1,-1.414,-1,0,0,0,1,1.414,1]
# iSobelSharp = SuanSharp(image,SobelX,SobelY)
# iPrewittSharp = SuanSharp(image,PrewittX,PrewittY)
# iIsotropicSharp = SuanSharp(image,IsotropicX,IsotropicY)

if __name__ == '__main__':
    app = QApplication(sys.argv)
 
    # 显示窗口
    win = My_UI()
    win.show()
    # win.run()
    sys.exit(app.exec_())

