import cv2
      
                
import json

from commonsCRNet import get_model

# Custom Imports 
import numpy as np
import PIL.Image as Image

from torchvision import transforms
import matplotlib.pyplot as plt
import random



# Access commons
model = get_model()
# Standard RGB transform
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])





class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
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


        ret, frame = self.video.read()
        print(frame.shape)

        '''out video'''
        width = frame.shape[1] #output size
        height = frame.shape[0] #output size
     
     








        while True:
     
            try:
                ret, frame = cap.read()

                scale_factor = 0.5
                frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                ori_img = frame.copy()
                print("Tryyyyyyyyyyyyyyyyyy") 
            except:
                print("test end")
                cap.release()
                break
                
    
       
       
            frame = frame.copy()

            
            img = transform(frame)

            img = img.cpu()
            
            
            output = model(img.unsqueeze(0))
            prediction = int(output.detach().cpu().sum().numpy())
            x = random.randint(1,100000) 
            density = 'static/density_map'+str(x)+'.jpg' 
            plt.imsave(density, output.detach().cpu().numpy()[0][0]) 
                
            cv2.putText(frame, "Count:" + str(prediction), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           
     
  
   
            print("org_img") 
     

      

            org_img = 'static/org_img.jpg' 

    
            cv2.imwrite(org_img, frame)

#--------------------------------------------------
#--------------------------------------------------
            image_names=[]

            image_names = [org_img,density ]
            print( " image names", image_names) 
            images = []
            max_width = 0 # find the max width of all the images
            total_height = 0 # the total height of the images (vertical stacking)

            for name in image_names:
        # open all images and find their sizes
        

                images.append(   cv2.resize( cv2.imread(name), (1200,450)  )  )
          #  images.append(cv2.imread(name))
                if images[-1].shape[1] > max_width:
                    max_width = images[-1].shape[1]
                total_height += images[-1].shape[0]

        # create a new array with a size large enough to contain all the images
            final_image = np.zeros((total_height,max_width,3),dtype=np.uint8)

            current_y = 0 # keep track of where your current image was last placed in the y coordinate
            for image in images:
                # add an image to the final array and increment the y coordinate
                final_image[current_y:image.shape[0]+current_y,:image.shape[1],:] = image
                current_y += image.shape[0]


            print("final_image", final_image) 
      









        
       
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
            ret, jpeg = cv2.imencode('.jpg', final_image)
        
        
        
        
            return jpeg.tobytes()
