"""
 @Usage: generate custom voc-format-dataset labels, convert .xml to .txt for each image
 @author: sun qian
 @date: 2019/9/25
 @note: dataset file structure must be modified as:
 --VOCdevkit
   --VOC2012
     --Annotations
     --ImageSets
        --Main (include train.txt, test.txt, val.txt)
     --JPEGImages
     --labels
 @ merge val and test: Run command: type 2012_test.txt 2012_val.txt  > test.txt
"""
import xml.etree.ElementTree as ET
import os
from os import getcwd

# file list - train.txt, test.txt, val.txt
sets = ['train', 'val']

# class name
classes = ["face"]


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('Annotations/%s.xml' % (image_id))
    out_file = open('labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':
    wd = getcwd()
    for image_set in sets:
        if not os.path.exists('labels/'):
            os.makedirs('labels/')
        image_ids = open('ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
        list_file = open('%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            line = '%s/images/%s.jpg\n' % (wd, image_id)
            list_file.write(line.replace("\\", '/'))
            convert_annotation(image_id)
        list_file.close()
    print("completed...")
