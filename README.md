# YOLO-FaceV2

### Introduction
YOLO-FaceV2: A Scale and Occlusion Aware Face Detector

### Framework Structure


### Requirments
Create a Python Virtual Environment.   
`conda create -n {name} python=x.x`
   
Enter Python Virtual Environment.   
`conda activate {name}`
   
Install pytorch in *[this](https://pytorch.org/get-started/previous-versions/)*.   
`pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html`
   
Install other python package.   
`pip install -r requirements.txt`
   
### Step-Through Example
#### Downloaded Dataset
`bash data/scripts/get_widerface.sh`

#### Dataset
```shell
python3 data/convert.py
python3 data/voc_label.py
```

#### Training
```shell
python train.py --weights preweight.pt   
                --data data/WIDER_FACE.yaml --cfg models/yolov5s_v2_RFEM_MultiSEAM.yaml    --batch-size 32 --epochs 250
```

#### Evaluate   
```shell
python widerface_pred.py --weights runs/train/x/weights/best.pt   
                        --save_folder ./widerface_evaluate/widerface_txt_x
cd widerface_evaluate/
python evaluation.py --pred ./widerface_txt_x
```

### Reference
*[]()*
*[https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)*