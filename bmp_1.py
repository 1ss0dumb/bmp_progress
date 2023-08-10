# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 22:37:48 2021

@author: WANGCY
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
# import cv2 as cv
from struct import unpack
def byte_to_int(str1):
    #从一个str类型的byte到int
    result=0
    for i in range(len(str1)):
        y=int(str1[len(str1)-1-i])
        result+=y*2**i
    return result

def breakup_byte(num1,n):
    #byte为输入的类型为byte的参数,n为每个数要的位数
    result=[]#返回的数字
    num=num1[2:]
    num_len=len(num)
    str1 = ""
    for i in range(8-num_len):
        str1+=str(0)
    num=str1+num
    for i in range(int(8/n)):
        temp=num[8-n*(i+1):8-n*i]
        result.append(byte_to_int(temp))
    result.reverse()
    return result

def breakup_16byte(str1,str2):
    #16位采用小端方式储存
    num1=str1[2:]
    num2=str2[2:]
    str1_ = ""
    str2_ = ""
    num_len1=len(num1)
    num_len2=len(num2)
    for i in range(8-num_len1):
        str1_+=str(0)
    num1=str1_+num1
    for i in range(8-num_len2):
        str2_+=str(0)
    num2=str2_+num2
    num = num2 + num1
    #16位用两个字节表示rgb设为555最后一个补0
    result = []
    r=byte_to_int(num[1:6])
    g=byte_to_int(num[6:11])
    b=byte_to_int(num[11:16])
    result.append(r*8)
    result.append(g*8)
    result.append(b*8)
    return result

def bmp_img_read_hist(path):
    #xxx=4
    #列出1,4,8,16,24图的位置
    # imgs=os.listdir(filename)
    #生成图片的路径保存在imgs_path
    img_path=path
    #print(imgs)
    # for img_name in imgs:
    #     img_path=filename+os.sep+img_name
    #     imgs_path.append(img_path)
    #执行
    # for img_path in imgs_path:
    img=open(img_path,"rb")
    #跳过bmp文件信息的开头，直接读取图片的size信息
    img.seek(28)
    bit_num=unpack("<i",img.read(4))[0]#字节数
    img.seek(10)
    #从开头到图片数据要的字节数
    to_img_data=unpack("<i",img.read(4))[0]
    img.seek(img.tell()+4)
    #unpack转为十进制
    img_width=unpack("<i",img.read(4))[0]
    img_height = unpack("<i", img.read(4))[0]
    img.seek(50)
    #颜色索引数
    color_num = unpack("<i", img.read(4))[0]
    #1位每个像素一位，4位一个像素0.5字节，8位一个像素1字节，16位一个像素2字节（555+0），24位一个像素3字节（bgr+alpha）
    #读取指针总共跳过54位到颜色盘,其中16,24位图像不需要调色盘
    img.seek(54)
    if(bit_num<=8):
        #多少字节调色板颜色就有2^n个
        color_table_num=2**int(bit_num)
        color_table=np.zeros((color_table_num,3),dtype=np.int)
        for i in range(color_table_num):
            b=unpack("B",img.read(1))[0];
            g = unpack("B", img.read(1))[0];
            r = unpack("B", img.read(1))[0];
            alpha=unpack("B", img.read(1))[0];
            color_table[i][0] = r;
            color_table[i][1] = g;
            color_table[i][2] = b;
    #将数据存入numpy中
    img.seek(to_img_data)
    img_np=np.zeros((img_height,img_width,3),dtype=np.int)
    num=0#计算读入的总字节数
    #数据排列从左到右，从下到上
    x=0
    y=0
    while y<img_height:
        while(x<img_width):
            if (bit_num <= 8):#小于等于8位的图像读取
                img_byte= unpack("B", img.read(1))[0]
                img_byte=bin(img_byte)
                color_index=breakup_byte(img_byte,bit_num)
                num+=1
                for index in color_index:
                    if(x<img_width):
                        img_np[img_height-y-1][x]=color_table[index]
                        x+=1
            elif(bit_num==24):#24位的图像读取
                num+=3
                g=unpack("B", img.read(1))[0]
                b=unpack("B", img.read(1))[0]
                r=unpack("B", img.read(1))[0]
                img_np[img_height - y - 1][x]=[r,b,g]
                x+=1
            elif (bit_num==16):#16位图像读取
                str1=bin(unpack("B", img.read(1))[0])
                str2=bin(unpack("B", img.read(1))[0])
                bgr_color=breakup_16byte(str1,str2)
                img_np[img_height - y - 1][x]=[bgr_color[0],bgr_color[1],bgr_color[2]]
                num+=2
                x+=1
        x=0
        y+=1
        while (num % 4 != 0):  # 每一行的位数都必须为4的倍数
            num += 1
            img.read(1)
        num=0
    return img_np,img_height,img_width,bit_num,img_path


def bmp_img_read_gray(path):
    #xxx=4
    #列出1,4,8,16,24图的位置
    # imgs=os.listdir(filename)
    #生成图片的路径保存在imgs_path
    img_path=path
    #print(imgs)
    # for img_name in imgs:
    #     img_path=filename+os.sep+img_name
    #     imgs_path.append(img_path)
    #执行
    # for img_path in imgs_path:
    img=open(img_path,"rb")
    #跳过bmp文件信息的开头，直接读取图片的size信息
    img.seek(28)
    bit_num=unpack("<i",img.read(4))[0]#字节数
    img.seek(10)
    #从开头到图片数据要的字节数
    to_img_data=unpack("<i",img.read(4))[0]
    img.seek(img.tell()+4)
    #unpack转为十进制
    img_width=unpack("<i",img.read(4))[0]
    img_height = unpack("<i", img.read(4))[0]
    img.seek(50)
    #颜色索引数
    color_num = unpack("<i", img.read(4))[0]
    #1位每个像素一位，4位一个像素0.5字节，8位一个像素1字节，16位一个像素2字节（555+0），24位一个像素3字节（bgr+alpha）
    #读取指针总共跳过54位到颜色盘,其中16,24位图像不需要调色盘
    img.seek(54)
    if(bit_num<=8):
        #多少字节调色板颜色就有2^n个
        color_table_num=2**int(bit_num)
        color_table=np.zeros((color_table_num,3),dtype=np.int)
        for i in range(color_table_num):
            b=unpack("B",img.read(1))[0];
            g = unpack("B", img.read(1))[0];
            r = unpack("B", img.read(1))[0];
            alpha=unpack("B", img.read(1))[0];
            gray=(r*30 + g*59 + b*11 ) / 100
            color_table[i][0]=gray;
            color_table[i][1] = gray;
            color_table[i][2] = gray;
    #将数据存入numpy中
    img.seek(to_img_data)
    img_np=np.zeros((img_height,img_width,3),dtype=np.int)
    num=0#计算读入的总字节数
    #数据排列从左到右，从下到上
    x=0
    y=0
    while y<img_height:
        while(x<img_width):
            if (bit_num <= 8):#小于等于8位的图像读取
                img_byte= unpack("B", img.read(1))[0]
                img_byte=bin(img_byte)
                color_index=breakup_byte(img_byte,bit_num)
                num+=1
                for index in color_index:
                    if(x<img_width):
                        img_np[img_height-y-1][x]=color_table[index]
                        x+=1
            elif(bit_num==24):#24位的图像读取
                num+=3
                g=unpack("B", img.read(1))[0]
                b=unpack("B", img.read(1))[0]
                r=unpack("B", img.read(1))[0]
                img_np[img_height - y - 1][x]=[r,b,g]
                x+=1
            elif (bit_num==16):#16位图像读取
                str1=bin(unpack("B", img.read(1))[0])
                str2=bin(unpack("B", img.read(1))[0])
                bgr_color=breakup_16byte(str1,str2)
                img_np[img_height - y - 1][x]=[bgr_color[0],bgr_color[1],bgr_color[2]]
                num+=2
                x+=1
        x=0
        y+=1
        while (num % 4 != 0):  # 每一行的位数都必须为4的倍数
            num += 1
            img.read(1)
        num=0
    return img_np,img_height,img_width,bit_num,img_path

def bmp_img_show(img_np,img_height,img_width,bit_num,img_path):
    plt.imshow(img_np)
    plt.show()
    #将图片以jpg格式保存在saved_img文件夹中
    # img_name_save="saved_img"+os.sep+"saved_"+img_path.split(os.sep)[1]
    # matplotlib.image.imsave(img_name_save, img_np.astype(np.uint8))
    #绘制直方图
    if bit_num<=8:
        plt.figure("hist")
        arr = img_np.flatten()
        plt.hist(arr, bins=2**bit_num,facecolor='green', alpha=0.75)
        plt.show()
    else:
        plt.figure("hist1")
        ar = np.array(img_np[:,:,0]).flatten()
        plt.hist(ar, bins=256,facecolor='r', edgecolor='r',alpha=0.5)
        plt.show()
        plt.figure("hist2")
        ag = np.array(img_np[:,:,1]).flatten()
        plt.hist(ag, bins=256, facecolor='g', edgecolor='g',alpha=0.5)
        plt.show()
        plt.figure("hist3")
        ab = np.array(img_np[:,:,2]).flatten()
        plt.hist(ab, bins=256, facecolor='b', edgecolor='b',alpha=0.5)
        plt.show()
    #将图片像素保存到txt文件中,由于numpy中的savetxt只能保存一维或者二维的数组，因此现将img_np展开
    # txt_name="img_txt"+'/'+"txt_"+(img_path.split(os.sep)[1]).split('.')[0]+'.txt'
    # img_np=np.reshape(img_np,(img_height*3,img_width))
    # np.savetxt(txt_name,img_np)
            
        
