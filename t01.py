# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:00:15 2017

@author: Administrator
MTCNN是基于深度学习的人脸检测方法，对自然环境中光线，角度和人脸表情变化更具
有鲁棒性，人脸检测效果更好；同时，内存消耗不大，可以实现实时人脸检测。
基于 facenet 
"""

from scipy import misc  
import tensorflow as tf
  
from src.align import detect_face  
import cv2  
import matplotlib.pyplot as plt  
 
  
minsize = 20 # minimum size of face  
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold  
factor = 0.709 # scale factor  
gpu_memory_fraction=1.0  
  
  
print('Creating networks and loading parameters')  
  
with tf.Graph().as_default():  
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)  
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))  
        with sess.as_default():  
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)  
  
image_path = 'data/images/Anthony_Hopkins_0001.jpg'              
  
img = misc.imread(image_path)              
bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)  
nrof_faces = bounding_boxes.shape[0]#人脸数目  
print('找到人脸数目为：{}'.format(nrof_faces))  
  
print(bounding_boxes)  
  
crop_faces=[]  
for face_position in bounding_boxes:  
    face_position=face_position.astype(int)  
    print(face_position[0:4])  
    cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)  
    crop=img[face_position[1]:face_position[3],  
             face_position[0]:face_position[2],]  
      
    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )  
    print(crop.shape)  
    crop_faces.append(crop)  
    plt.imshow(crop)  
    plt.show()  
      
plt.imshow(img)  
plt.show()  