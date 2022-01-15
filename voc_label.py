import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["face_mask","face"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):

    in_file = open('VOCdevkit/Annotations/%s.xml'%(image_id))
    out_file = open('VOCdevkit/labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()

    # size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)

    image = cv2.imread('VOCdevkit/JPEGImages/%s.jpg'%(image_id))
    h,w = image.shape[0],image.shape[1]

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')



if __name__ == '__main__':
    with open('VOCdevkit/ImageSets/Main/pscalvoc.txt',"r") as fread:
        image_ids = fread.readlines()

        # print(image_ids)

    list_file = open('new_data.txt', 'w')
    for image_id in image_ids:
        image_id = image_id.strip()
        list_file.write('VOCdevkit/JPEGImages/%s.jpg\n'%(image_id))
        convert_annotation(image_id)
    list_file.close()

