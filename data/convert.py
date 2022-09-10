# -*- coding: utf-8 -*-

import shutil
import random
import os
import string
from skimage import io

headstr = """\
<annotation>
    <folder>VOC2012</folder>
    <filename>%06d.jpg</filename>
    <source>
        <database>My Database</database>
        <annotation>PASCAL VOC2012</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''




def writexml(idx, head, bbxes, tail):
    filename = ("Annotations/%06d.xml" % (idx))
    f = open(filename, "w")
    f.write(head)
    for bbx in bbxes:
        f.write(objstr % ('face', bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]))
    f.write(tail)
    f.close()


def clear_dir():
    if shutil.os.path.exists(('Annotations')):
        shutil.rmtree(('Annotations'))
    if shutil.os.path.exists(('ImageSets')):
        shutil.rmtree(('ImageSets'))
    if shutil.os.path.exists(('JPEGImages')):
        shutil.rmtree(('JPEGImages'))

    shutil.os.mkdir(('Annotations'))
    shutil.os.makedirs(('ImageSets/Main'))
    shutil.os.mkdir(('JPEGImages'))


def excute_datasets(idx, datatype):

    f = open(('ImageSets/Main/' + datatype + '.txt'), 'a')
    f_bbx = open(('wider_face_split/wider_face_' + datatype + '_bbx_gt.txt'), 'r')

    while True:
        filename = f_bbx.readline().strip('\n')

        if not filename:
            break


        im = io.imread(('WIDER_' + datatype + '/images/' + filename))
        head = headstr % (idx, im.shape[1], im.shape[0], im.shape[2])
        nums = f_bbx.readline().strip('\n')
        bbxes = []
        if nums=='0':
            bbx_info= f_bbx.readline()
            continue
        for ind in range(int(nums)):
            bbx_info = f_bbx.readline().strip(' \n').split(' ')
            bbx = [int(bbx_info[i]) for i in range(len(bbx_info))]
            # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
            if bbx[7] == 0:
                bbxes.append(bbx)
        writexml(idx, head, bbxes, tailstr)
        shutil.copyfile(('WIDER_' + datatype + '/images/' + filename), ('JPEGImages/%06d.jpg' % (idx)))
        f.write('%06d\n' % (idx))
        idx += 1
    f.close()
    f_bbx.close()
    return idx


if __name__ == '__main__':
    clear_dir()
    idx = 1
    idx = excute_datasets(idx, 'train')
    idx = excute_datasets(idx, 'val')
    print('Complete...')
