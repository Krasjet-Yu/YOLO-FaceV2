# YOLO-FaceV2

### Introduction
YOLO-FaceV2: A Scale and Occlusion Aware Face Detector     
*[Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S0031320324004655)* or *[arxiv](https://arxiv.org/abs/2208.02019)*   


### Framework Structure
![](data/images/yolo-facev2.jpg)

### Requirments
Create a Python Virtual Environment.   
```shell
conda create -n {name} python=x.x
```

Enter Python Virtual Environment.   
```shell
conda activate {name}
```

Install pytorch in *[this](https://pytorch.org/get-started/previous-versions/)*.  
```shell 
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other python package.   
```shell
pip install -r requirements.txt
```
   
### Step-Through Example
#### Downloaded Dataset
```shell
bash data/scripts/get_widerface.sh
```

#### Dataset
Download the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset. Then convert it to YOLO format.
```shell
# You can modify convert.py and voc_label.py if needed.
python3 data/convert.py
python3 data/voc_label.py
```

```shell
cd data
python3 train2yolo.py /path/to/original/widerface/train [/path/to/save/widerface/train]
python3 val2yolo.py  /path/to/original/widerface [/path/to/save/widerface/val]
```


## Preweight
| Model            | Easy  | Medium | Hard  | PreWeight  |
|------------------|-------|--------|-------|------------|
| YOLO-Facev2n     | 0.936 | 0.921  | 0.812 | [yolo-facev2n.pt](https://github.com/Krasjet-Yu/YOLO-FaceV2/releases/download/v2.1.0/yolo-facev2m-preweight.pt) |
| YOLO-FaceV2s     | 0.983 | 0.970  | 0.893 | [yolo-facev2s.pt](https://github.com/Krasjet-Yu/YOLO-FaceV2/releases/download/v2.1.0/yolo-facev2s-preweight.pt) |
| YOLO-FaceV2m     | 0.984 | 0.973  | 0.905 | [yolo-facev2m.pt](https://github.com/Krasjet-Yu/YOLO-FaceV2/releases/download/v2.1.0/yolo-facev2m-preweight.pt) |
| **YOLO-FaceV2l** | 0.986 | 0.979  | 0.919 | [yolo-facev2l.pt](https://github.com/Krasjet-Yu/YOLO-FaceV2/releases/download/v2.1.0/yolo-facev2l-preweight.pt) |


#### Training
```shell
python train.py --weights preweight.pt    
                --data data/WIDER_FACE.yaml    
                --cfg models/yolov5s_v2_RFEM_MultiSEAM.yaml     
                --batch-size 32   
                --epochs 250
```


#### Evaluate   
```shell
python3 test_widerface.py --weights 'your test model' --img-size 640
  
cd widerface_evaluate/    
python evaluation.py --pred ./widerface_txt_x
```
<div style="display: flex; gap: 10px;">
  <img src="assert/easy.jpg" alt="Easy" style="width: 260px;"/>
  <img src="assert/medium.jpg" alt="Medium" style="width: 260px;"/>
  <img src="assert/hard.jpg" alt="Hard" style="width: 260px;"/>
</div>


### Visualize
Visualization of small-scale face detection effect:

<p align = "center">    
<img  src="assert/scale.jpg" width="350" />
</p>

Visualization of the heat map effect of attention covering human faces:

<p align = "center">    
<img  src="assert/occlusion.jpg" width="350" />
</p>


### Finetune
see in *[https://github.com/ultralytics/yolov5/issues/607](https://github.com/ultralytics/yolov5/issues/607)*
```shell
# Single-GPU
python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --evolve

# Multi-GPU
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting GPU '$i'...' &&
  nohup python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --device $i --evolve > evolve_gpu_$i.log &
done

# Multi-GPU bash-while (not recommended)
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting GPU '$i'...' &&
  "$(while true; do nohup python train.py... --device $i --evolve 1 > evolve_gpu_$i.log; done)" &
done
```

### Reference
*[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)*    
*[https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)*   
*[https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)*   
*[https://github.com/dongdonghy/repulsion_loss_pytorch](https://github.com/dongdonghy/repulsion_loss_pytorch)*   

### Cite

If you think this work is helpful for you, please cite

```shell
@article{YU2024110714,
title = {YOLO-FaceV2: A scale and occlusion aware face detector},
journal = {Pattern Recognition},
pages = {110714},
year = {2024},
issn = {0031-3203},
url = {https://www.sciencedirect.com/science/article/pii/S0031320324004655},
author = {Ziping Yu and Hongbo Huang and Weijun Chen and Yongxin Su and Yahui Liu and Xiuying Wang},
keywords = {Face detection, YOLO, Scale-aware, Occlusion, Imbalance problem},
}

```

### Contact

We use code's license is MIT License. The code can be used for business inquiries or professional support requests.
