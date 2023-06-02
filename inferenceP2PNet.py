import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, Event
from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser


parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

print(args)




def get_prediction_webcam(event: Event):  
    print("start p2pnet")
    
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    
    # move to GPU
    model.to(device)
    
    
    # load trained model
    #using Args
    """
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    """
    #Loading file directly
    checkpoint = torch.load(Path('/home/zaki/Documents/Master/Code/image/P2PNet/CrowdCounting-P2PNet-main(mycode)/weights/SHTechA.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])


    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your image path here
    #img_path = "./vis/demo1.jpg"
    





    '''in video'''
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')


    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    print(frame.shape)

    '''out video'''

    scale_factor = 0.4

    width = frame.shape[1] #output size
    height = frame.shape[0] #output size
    #out = cv2.VideoWriter('./demo.avi', fourcc, 30, (width, height))
    #out = cv2.VideoWriter('./demo.avi', fourcc, 30, (1280, 1280))

    while True:
        try:
            ret, frame = cap.read()


            new_width = width // 128 * 128
            new_height = height // 128 * 128
            
            #frame = imutils.resize(frame,width=int(new_width),height=int(new_height))


            scale_factor = 0.4
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            img_raw= frame.copy()
            ori_img = frame.copy()
            
        
        except:
            print("Test End")
            cap.release()
            break
            


        frame = frame.copy()

        # pre-proccessing
        img = transform(frame)
        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)

        with torch.no_grad():

            # run inference
            outputs = model(samples)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

            outputs_points = outputs['pred_points'][0]

            threshold = 0.5
            # filter the predictions
            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
            predict_cnt = int((outputs_scores > threshold).sum())

            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]


            outputs_points = outputs['pred_points'][0]
            print("Number of persons in the picture is: ",predict_cnt)

            # draw the predictions
            size = 2
            
            img_to_draw = img_raw
            
            for p in points:
                img_to_draw = cv2.circle(img_raw , (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        
            
            
            #res = np.vstack((ori_img, img_to_draw))

            #cv2.putText(res, "Count:" + str(predict_cnt), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        
            # save the visualized image
            #cv2.imwrite('./demo.jpg', res)
            '''write in out_video'''
            #res = cv2.resize(res, (1280,1280))
            #out.write(res)

            
            img_to_draw = cv2.resize(img_to_draw, (1500,720))
            cv2.putText(img_to_draw, "Count:" + str(predict_cnt), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("dst",img_to_draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break
               
            if event.is_set():
               print('The thread was stopped prematurely.')
               cv2.destroyAllWindows()
               break



 
 




def get_prediction(file):  
 
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    
    # move to GPU
    model.to(device)
    
    
    # load trained model
    #using Args
    """
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    """
    #Loading file directly
    checkpoint = torch.load(Path('/home/zaki/Documents/Master/Code/image/P2PNet/CrowdCounting-P2PNet-main(mycode)/weights/SHTechA.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])


    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your image path here
    #img_path = "./vis/demo1.jpg"
    
    if (file.endswith(".mp4")):
        '''in video'''
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')


        cap = cv2.VideoCapture(file)
        ret, frame = cap.read()
        print(frame.shape)

        '''out video'''

        scale_factor = 0.4

        width = frame.shape[1] #output size
        height = frame.shape[0] #output size
        #out = cv2.VideoWriter('./demo.avi', fourcc, 30, (width, height))
        out = cv2.VideoWriter('./demo.avi', fourcc, 30, (1280, 1280))

        while True:
            try:
                ret, frame = cap.read()


                new_width = width // 128 * 128
                new_height = height // 128 * 128
            
                #frame = imutils.resize(frame,width=int(new_width),height=int(new_height))


                scale_factor = 0.4
                frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                img_raw= frame.copy()
                ori_img = frame.copy()
            
        
            except:
                print("Test End")
                cap.release()
                break
            


                frame = frame.copy()

            # pre-proccessing
            img = transform(frame)
            samples = torch.Tensor(img).unsqueeze(0)
            samples = samples.to(device)

            with torch.no_grad():

                # run inference
                outputs = model(samples)
                outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

                outputs_points = outputs['pred_points'][0]

                threshold = 0.5
               # filter the predictions
                points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
                predict_cnt = int((outputs_scores > threshold).sum())

                outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]


                outputs_points = outputs['pred_points'][0]
                print("Number of persons in the picture is: ",predict_cnt)

                # draw the predictions
                size = 2

                for p in points:
                    img_to_draw = cv2.circle(img_raw , (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
                res = np.vstack((ori_img, img_to_draw))

                cv2.putText(res, "Count:" + str(predict_cnt), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # save the visualized image
                #cv2.imwrite('./demo.jpg', res)
                '''write in out_video'''
                #res = cv2.resize(res, (1280,1280))
                #out.write(res)
            
                cv2.putText(img_to_draw, "Count:" + str(predict_cnt), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("dst",img_to_draw)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   break


    else:
        # load the images
        img_raw = Image.open(file).convert('RGB')

        # round the size
        width, height = img_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    
        ori_imge  = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        # pre-proccessing
        img = transform(img_raw)

        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)
        # run inference
        outputs = model(samples)
    
    
    
        prediction = outputs['pred_points'][0].detach().cpu().numpy()
        prediction = np.resize(prediction, (width, height))
    
    
        print("prediction",prediction)
        print("prediction Shape",prediction.shape)
    
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        threshold = 0.5
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]



        outputs_points = outputs['pred_points'][0]
    
        print("Number of persons in the picture is: ",predict_cnt)

        # draw the predictions
        size = 2
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    

    
    
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        res = np.vstack((ori_imge, img_to_draw))
    
        x = random.randint(1,100000)
        density = 'static/density_map'+str(x)+'.jpg' 
        #plt.imsave(density, prediction)
    
        cv2.imwrite(density, img_to_draw)
    
        return predict_cnt , density
 
 
 
 
    
    
