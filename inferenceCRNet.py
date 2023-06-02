import json

from commonsCRNet import get_model

# Custom Imports 
import numpy as np
import PIL.Image as Image
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from threading import Thread, Event


# Access commons
model = get_model()
# Standard RGB transform
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])




    
def get_prediction_webcam(event: Event ):


    print("event_isSet In Inference: ",                 event.is_set()) 
 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    print(frame.shape)

    '''out video'''
    width = frame.shape[1] #output size
    height = frame.shape[0] #output size
    out = cv2.VideoWriter('./demo.avi', fourcc, 30, (width, height))

    while True:
        print("event_isSet In Inference bellow While true: ",                 event.is_set()) 
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
      

#-----------------------------------------------
#--------------------
        


        cv2.imshow("dst",final_image)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
        
        
        print("event_isSet In Inference above if: ",                 event.is_set()) 
        
        
        if event.is_set():
            print('The thread was stopped prematurely.')
            #cv2.destroyWindow("dst"+str(y))
            #event.clear() 
           # cv2.waitKey(0)
            cap.release()
            cv2.destroyAllWindows()
            break





def get_prediction(file):



        
    print("------------------------------------")    
    print(file)
    if (file.endswith(".mp4")):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        cap = cv2.VideoCapture(file)
        ret, frame = cap.read()
        print(frame.shape)

        '''out video'''
        width = frame.shape[1] #output size
        height = frame.shape[0] #output size
        out = cv2.VideoWriter('./demo.avi', fourcc, 30, (width, height))

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
            #image = tensor_transform(frame)
            #image = img_transform(image).unsqueeze(0)
            #image = image.to(device)
            
            img = transform(frame)

            img = img.cpu()
            
            
            output = model(img.unsqueeze(0))
            prediction = int(output.detach().cpu().sum().numpy())
            x = random.randint(1,100000) 
            density = 'static/density_map'+str(x)+'.jpg' 
            plt.imsave(density, output.detach().cpu().numpy()[0][0]) 
                
            cv2.putText(frame, "Count:" + str(prediction), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           
            #img1 = cv2.convertScaleAbs(output.detach().cpu().numpy()[0][0], alpha=(255.0))
            #cv2.imshow("dst",img1)
      



   

            org_img = 'static/org_img.jpg' 

    
            cv2.imwrite(org_img, frame)

#--------------------------------------------------
#--------------------------------------------------


            image_names = [org_img,density ]
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



      

#-----------------------------------------------
#--------------------
       
          


            cv2.imshow("dst",final_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            
            print("event_isSet In Inference: ",                 event.is_set()) 
            if event.is_set():
                print('The thread was stopped prematurely.')
                cv2. destroyAllWindows()
                break
                   
                   
    else :




        img = transform(Image.open(file).convert('RGB'))
        img = img.cpu()
        output = model(img.unsqueeze(0))
        prediction = int(output.detach().cpu().sum().numpy())
        x = random.randint(1,100000) 
        density = 'static/density_map'+str(x)+'.jpg' 
        plt.imsave(density, output.detach().cpu().numpy()[0][0]) 
    
    
    
    
    return prediction, density
