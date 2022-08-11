# YOLO-FaceV2

#### 介绍
YOLO-FaceV2: A Scale and Occlusion Aware Face Detector

#### 软件架构
软件架构说明


#### 安装教程

1.  xxxx
2.  xxxx
3.  xxxx

#### 使用说明

1.  train
python train.py --weights preweight.pt --data data/WIDER_FACE.yaml --cfg models/yolov5s_v2_RFEM_MultiSEAM.yaml  --batch-size 16 --epochs 100
2.  evaluate
python widerface_pred.py --weights runs/train/x/weights/best.pt --save_folder ./widerface_evaluate/widerface_txt_x
cd widerface_evaluate/
python evaluation.py --pred ./widerface_txt_x

3.  xxxx

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request
