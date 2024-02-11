'''
https://discuss.pytorch.org/t/interpreting-output-of-backward-hook/114357/3
https://www.jianshu.com/p/69e57e3526b3
'''
from turtle import color
import cv2
import time
import argparse
from matplotlib.colors import same_color
import numpy as np
from models.gradcam import YOLOV5GradCAM
from models.yolo_detector import YOLOV5TorchObjectDetector
from pathlib import Path


def get_res_img(bbox, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET).astype(np.float32)
    
    bbox = [int(b) for b in bbox]
    tmp = np.ones_like(res_img,dtype=np.float32) * 0
    tmp[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
    # heatmap = cv2.multiply(heatmap, tmp).astype(np.float32)

    res_img = cv2.add(res_img, heatmap)
    res_img = (res_img / res_img.max())
    return res_img, heatmap


def put_text_box(bbox, cls_name, res_img, thickness=2):
    x1, y1, x2, y2 = [int(b) for b in bbox]
    res_img = cv2.rectangle(res_img, (x1, y1), (x2, y2), (0,0,255), thickness=2)
    w, h = cv2.getTextSize(cls_name, 0, fontScale=thickness, thickness=2)[0]  # text width, height
    outside = y1 - h - 3 >= 0  # label fits outside box
    t0, t1 = x1, y1 - 2 if outside else y1 + h + 2
    res_img = cv2.putText(res_img, cls_name, (t0, t1), color=(0,0,225), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, lineType=cv2.LINE_AA)
    return res_img


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="yolov5s.pt", help='Path to the model')
    parser.add_argument('--img-path', type=str, default='images/dog.jpg', help='input image path')
    parser.add_argument('--output-dir', type=str, default='outputs', help='output dir')
    parser.add_argument('--img-size', type=int, default=640, help="input image size")
    parser.add_argument('--target-layer', type=str, default='model_23_conv',
                        help='The layer hierarchical address to which gradcam will applied,'
                            ' the names should be separated by underline')
    parser.add_argument('--method', type=str, default='gradcam', help='gradcam or gradcampp')
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--visual-path', type=str, default='./outputs/visualization/', help='feature visualize path')
    parser.add_argument('--names', type=str, default=None,
                        help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')
    args = parser.parse_args()

    device = args.device
    input_size = (args.img_size, args.img_size)
    
    print('[INFO] Loading the model')
    model = YOLOV5TorchObjectDetector(args.model_path, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","))
    
    if args.method == 'gradcam':
        saliency_method = YOLOV5GradCAM(model=model, layer_name=args.target_layer, img_size=input_size)
    
    img_path = Path.cwd() / args.img_path
    img = cv2.imread(str(img_path))
    torch_img = model.preprocessing(img[..., ::-1])
    
    tic = time.time()
    masks, logits, [boxes, _, class_names, _] = saliency_method(torch_img)
    print('bbbooooox', type(class_names[0]))
    print("total time:", round(time.time() - tic, 4))
    result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # convert to bgr
    save_path = Path.cwd() / args.output_dir / '{}'.format(img_path.stem)
    if not save_path.exists():
        save_path.mkdir()
    
    for i, mask in enumerate(masks):
        res_img = result.copy()
        bbox, cls_name = boxes[0][i], class_names[0][i]
        print(mask.shape)
        res_img, heatmat = get_res_img(bbox, mask, res_img)
        color_img = (res_img * 255).astype(np.uint8)
        # color_img = put_text_box(bbox, cls_name, color_img)
        cv2.imwrite(str(save_path / '{0}_{1}.jpg'.format(img_path.stem, i)), color_img)

if __name__ == '__main__':
    main()