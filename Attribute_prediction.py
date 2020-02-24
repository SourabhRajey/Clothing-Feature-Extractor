# make a prediction for a new image.
from __future__ import division
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import os
import sys
import cv2
import requests
import json							
import numpy as np
import pandas as pd

from color_extractor import ImageToColor

from fastai import *
from fastai.vision import *


labelsPath = 'mscoco_label_map.pbtxt'
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
weightsPath = 'frozen_inference_graph.pb'
configPath = 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
pattern_model = load_model('pattern.h5')
neck_model = load_model('neckline_4cat_5.h5')



npz = np.load('color_names.npz')
img_to_color = ImageToColor(npz['samples'], npz['labels'])
def load_image(filename):
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	img = img_to_array(img)
	img = img.reshape(1, 28, 28, 1)
	img = img.astype('float32')
	img = img / 255.0
	return img


def canny_grab(image):
	canny_out = cv2.Canny(image,0,image.flatten().mean())
	y,x = canny_out.nonzero()
	top_left = x.min(), y.min()
	bot_right = x.max(), y.max()
	return canny_out, top_left, bot_right


def grab_cut(image, top_left, bot_right):
	mask = np.zeros(image.shape[:2],np.uint8)
	background = np.zeros((1,65),np.float64)
	foreground = np.zeros((1,65),np.float64)
	roi = (top_left[0],top_left[1],bot_right[0],bot_right[1])
	cv2.grabCut(image, mask, roi, background, foreground, 5, cv2.GC_INIT_WITH_RECT)
	new_mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	image = image*new_mask[:,:,np.newaxis]
	return image


def bg_removal(orig, grabbed):
	kernel = np.ones((3,3),np.uint8)
	mean,std = cv2.meanStdDev(cv2.cvtColor(orig,cv2.COLOR_BGR2HLS), cv2.inRange(grabbed,0,0))
	min_thresh = mean - std
	max_thresh = mean + std
	grab_bg = cv2.inRange(cv2.cvtColor(grabbed,cv2.COLOR_BGR2HLS),min_thresh,max_thresh)
	dilated_bg = cv2.morphologyEx(grab_bg, cv2.MORPH_OPEN, kernel)
	return dilated_bg


def watershed(grabbed):
	gray = cv2.cvtColor(grabbed,cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=3)

	background = cv2.dilate(opening, kernel, iterations=2)

	dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
	_, foreground = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)

	foreground = np.uint8(foreground) 
	unknown = cv2.subtract(background,foreground)

	_, ccs = cv2.connectedComponents(foreground)
	ccs = ccs + 1
	ccs[unknown==255] = 0
	ccs = cv2.watershed(grabbed,ccs)
	return ccs


def segmentaion(img):
	canny_img, tl, br = canny_grab(img)
	roi = img[tl[1]:br[1], tl[0]:br[0]]
	i = 0
	grab = grab_cut(img, tl, br)
	bg_mask = bg_removal(img, grab)
	bg_removed = cv2.subtract(grab, cv2.cvtColor(bg_mask,cv2.COLOR_GRAY2BGR))
	grab = cv2.GaussianBlur(grab, (15,15),0)
	watershed_out = watershed(grab)
	final_piece = cv2.bitwise_and(img, cv2.cvtColor(cv2.inRange(watershed_out,1,1),cv2.COLOR_GRAY2BGR))
	gray = cv2.cvtColor(final_piece, cv2.COLOR_BGR2GRAY)
	img = cv2.medianBlur(gray,5)
	th1 = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	return final_piece

path = "Fine-Grained-Clothing-Classification/data/cloth_categories"

classes = ['Shirt', 'Blazer', 'Button-Down', 'Bomber', 'Anorak', 'Tee', 'Tank', 'Top', 'Sweater', 'Flannel', 'Hoodie', 'Cardigan', 'Jacket', 'Henley', 'Poncho', 'Jersey', 'Turtleneck', 'Parka', 'Peacoat', 'Halter', 'Skirt', 'Shorts', 'Jeans', 'Joggers', 'Sweatpants', 'Jeggings', 'Cutoffs', 'Sweatshorts', 'Leggings', 'Culottes', 'Chinos', 'Trunks', 'Sarong', 'Gauchos', 'Jodhpurs', 'Capris', 'Dress', 'Romper', 'Coat', 'Kimono', 'Jumpsuit', 'Robe', 'Caftan', 'Kaftan', 'Coverup', 'Onesie']
UpperWear = ['Tee','Top','Sweater','Tank']
single_img_data = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(),size=150).normalize(imagenet_stats)

learn = cnn_learner(single_img_data, models.resnet34, metrics=accuracy)

learn.load('stage-1_sz-150')

def pattern(frame):
	blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
	net.setInput(blob)
	(boxes, masks) = net.forward(["detection_out_final","detection_masks"])
	boxes = boxes[0][0][boxes[0][0][:,2].argsort()[::-1]]
	confidence = boxes[0, 2]
	(H, W) = frame.shape[:2]
	box = boxes[0, 3:7] * np.array([W, H, W, H])
	(startX, startY, endX, endY) = box.astype("int")
	frame1=frame[startY:endY, startX:endX]
	try:
		test_image=cv2.resize(frame1,(64,64),interpolation=cv2.INTER_AREA)
	except Exception as e:
		test_image=cv2.resize(frame,(64,64),interpolation=cv2.INTER_AREA)
	test_image = np.expand_dims(test_image, axis = 0)
	result_pattern = pattern_model.predict_classes(test_image)   
	pattern_classes = ['Floral','Graphics','Plaid','Solid','Spotted','Striped']            
	prediction_pattern = pattern_classes[int(result_pattern)]
	return str(prediction_pattern)
def NeckLine(img):
	y = 50
	x = 110
	h = 260
	w = 260
	# << number between 0 and 150
	
	img= cv2.resize(img,(480,736),interpolation=cv2.INTER_AREA)
	img=img[y:y+h, x:x+w]
	x_val=np.stack(img)
	x_val=np.reshape(x_val,[1,260,260,3])
	pred = np.around(neck_model.predict(x_val),2) 
	columns=['polo','round_neck','v_neck']
	predictions = sorted(zip(columns, map(float, pred[0])), key=lambda p: p[1], reverse=True)
	neckline,score = predictions[:1][0]
	return neckline

def run_example(img_path):
	# load the imagemask_rcnn_inception_v2_coco_2018_01_28
	#img_path = "/home/sourabhrajey/sourav/fashion-mnist/image/"+img_path
	img = cv2.imread(img_path)
	final_piece = segmentaion(img)
	cv2.imwrite('a.png',final_piece)
	img1 = cv2.cvtColor(final_piece, cv2.COLOR_BGR2RGB)
	#img=final_piece
	obj={}
	#obj["fcid"]=fcid
	_img_color = img_to_color.get(img1)
	obj["color"]=_img_color[0] 

	_,_,losses = learn.predict(open_image(img_path))

	predictions = sorted(zip(classes, map(float, losses)), key=lambda p: p[1], reverse=True)
	
	Type,Accuracy = predictions[:1][0]

	obj["type"]= Type
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	obj["pattern"]=pattern(img)
	if obj["type"] in UpperWear:
		obj["neckline"] = NeckLine(img)
	else:
		obj["neckline"] = "None"
	return obj

IMG_FILE_SRC =  "image/image18.jpg"
#img = cv2.imread(IMG_FILE_SRC)
data=run_example(IMG_FILE_SRC)
print(data)
'''
genderdict = {
    "Tee" : "male",
    'Top' : 'female',
    'Tank' : 'both',
    'Henley' : 'male',
    'Sweater' : 'both',
    'Button-Down' : 'male',
    'Shirt' : 'male',
    'Blouse' : "male",
    'Cardigan' : "male",
    'Jacket' : 'both',
    'Skirt' : 'female',
    'Chinos' : 'both',
    'Peacoat' : 'male',
    'Shorts' : 'both',
    'Jeans' : 'both',
    'Flannel' : 'male',
    'Leggings' : 'female',
    'Dress' : 'both',
    'Anorak' : 'both',
    'Parka' : 'both',
    'Bomber' : 'both'
}

url = "http://cdn.fcglcdn.com/brainbees/images/products/438x531/"
f = open("1.txt","r")
out = open("donefcids.txt","a+")
for x in f:
	fcid = x.strip()
	webp_imagepath="image/"+str(fcid)+"a.webp"
	png_path="image/"+str(fcid)+"a.png"
	os.system("curl "+url+str(fcid)+"a.webp"+" > "+webp_imagepath)
	os.system("dwebp "+webp_imagepath+" -o "+png_path)
	data = Attribute_prediction.run_example(png_path)
	gender = genderdict[data["type"]]
	payload={'fcid':fcid,'pattern':data["pattern"],'type':data["type"],'gender':gender,'necktype':data["neckline"]}
	#r = requests.post("http://10.0.0.218:9999/dumpdata",params=payload)
	os.system("rm "+webp_imagepath)
	os.system("rm "+png_path)
	f.write(fcid)
'''