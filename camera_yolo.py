import argparse
import os
import platform
import sys
from pathlib import Path

import random
import torch
import numpy as np
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, check_dataset, check_requirements,
                           check_yaml, clean_str, cv2, is_colab, is_kaggle, segments2boxes, unzip_file, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

device = select_device('')
weights='/home/abdou/Bureau/yolov5copy/best11.pt'
model = DetectMultiBackend(weights, device=device, dnn=False, data=ROOT /'data/coco128.yaml', fp16=False)


class VideoCamera(object):
    weights='/home/abdou/Bureau/yolov5copy/best11.py'  # model path or triton URL
    source=ROOT / 'data/images'  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
    imgsz=(640, 640)  # inference size (height, width)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_txt=False  # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    update=False  # update all models
    project=ROOT / 'runs/detect'  # save results to project/name
    name='exp'  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    dnn=False  # use OpenCV DNN for ONNX inference
    vid_stride=1  # video frame-rate stride
    def __init__(self,fileName):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        if (fileName ==''):
            self.video = cv2.VideoCapture(0)
        else:  
            self.video = cv2.VideoCapture(fileName)   
#        self.video = cv2.resize(self.video,(840,640))
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        
        cap =self.video 
      
      
      
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    #cap = cv2.VideoCapture(args.video_path)
    #cap= cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
 
        ret, frame = cap.read()
        print(frame.shape)

        '''out video'''
        width = frame.shape[1] #output size
        height = frame.shape[0] #output size
        out = cv2.VideoWriter('./demo.avi', fourcc, 30, (width, height))

        

    
        
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size((640,640), s=stride)  # check image size

        
        
            

        while True:
            try:
                ret, frame = cap.read()

                scale_factor = 0.5
                frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                ori_img = frame.copy()
            except:
                print("test end")
                cap.release()
                break
            frame = frame.copy()
            source = str(frame)

            bs = 1  # batch_size

            with torch.no_grad():

                # Run inference
                model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
                seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
                with dt[0]:
                    im = letterbox(frame, 640, stride=32, auto=True)[0]  # padded resize
                    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    im = np.ascontiguousarray(im)
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                    # Inference
                with dt[1]:
                    pred = model(im, augment=False, visualize=False)

                    # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                    # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    #p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    #p = Path(p)  # to Path
                    #s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    annotator = Annotator(frame, line_width=1,font_size=1, example=str(names))
                    n=0
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = f'{names[c]} {conf:.02f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))

                                
                    # Stream results
                    im0 = annotator.result()
                    if torch.is_tensor(n):
                        prediction = n.item()
                    else:
                        prediction = n
                    img_to_draw = cv2.resize(im0, (1500,720))
                    cv2.putText(img_to_draw, 'Number of people=' + str(prediction), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)	
                    #cv2.imshow('l' , np.array(im0, dtype = np.uint8 ) )
                    cv2.waitKey(25)
                        
                    res = img_to_draw
                    im0 = annotator.result()
                    if torch.is_tensor(n):
                        prediction = n.item()
                    else:
                        prediction = n
                        
                    res = img_to_draw

                        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
                ret, jpeg = cv2.imencode('.jpg', res)
        
        
        
                return jpeg.tobytes()


                

                
 
     
        
