import os
import os
import cv2
import csv
import argparse
from PIL import Image
from scipy.io import loadmat
import csv
import math
import random

import pandas as pd
import numpy as np




class DataLoader(object):
    def __init__(self, data_path, gt_path, transform=None,shuffle=False, gt_downsample=False):
     
        self.data_path = data_path
        self.gt_path = gt_path
        self.shuffle = shuffle
        self.transform=transform
        self.gt_downsample = gt_downsample
        self.data_files = [filename for filename in os.listdir(data_path)]
        self.num_samples = len(self.data_files)
        self.blob_list = []

        for fname in self.data_files:
            #img = cv2.imread(os.path.join(self.data_path, fname),0)
            img = cv2.imread(os.path.join(self.data_path, fname))
            img1 = cv2.imread(os.path.join(self.data_path, fname))
            img = img.astype(np.float32, copy=False)
            ht = img.shape[0]
            wd = img.shape[1]
            ht_1 = int((ht / 4) * 4)
            wd_1 = int((wd / 4) * 4)
            img = cv2.resize(img, (wd_1, ht_1))
            #img = img.reshape((img.shape[0], img.shape[1], 1)) 
            
            

            img = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] +'.csv'),header=None).values

            den = den.astype(np.float32, copy=False)
            if self.gt_downsample:
                wd_1 = int(wd_1 / 4)
                ht_1 = int(ht_1 / 4)
            den = cv2.resize(den, (wd_1, ht_1))
            den = den * ((wd * ht) / (wd_1 * ht_1))
            den = den.reshape((den.shape[0], den.shape[1], 1))

            blob = dict()
            blob['data'] = self.transform(img)
            blob['gt'] = den
            blob['fname'] = fname
            self.blob_list.append(blob)

        if self.shuffle:
            np.random.shuffle(self.blob_list)

    def flow(self, batch_size=32):
        loop_count = self.num_samples // batch_size
        while True:
            np.random.shuffle(self.blob_list)
            for i in range(loop_count):
                blobs = self.blob_list[i*batch_size: (i+1)*batch_size]
                X_batch = np.array([blob['data'] for blob in blobs])
                Y_batch = np.array([blob['gt'] for blob in blobs])
                yield X_batch, Y_batch

    def flow1(self, batch_size=32):
        loop_count = self.num_samples // batch_size
        while True:
            np.random.shuffle(self.blob_list)
            for i in range(loop_count):
                blobs = self.blob_list[i*batch_size: (i+1)*batch_size]
                X_batch = [blob['data'] for blob in blobs]
                Y_batch = np.array([blob['gt'] for blob in blobs])
                yield X_batch, Y_batch                
    
    def __len__(self):
        return self.num_samples
        

    def get_all(self):
        X = np.array([blob['data'] for blob in self.blob_list])
        Y = np.array([blob['gt'] for blob in self.blob_list])
        return X, Y

    def __iter__(self):
        for blob in self.blob_list:
            yield blob