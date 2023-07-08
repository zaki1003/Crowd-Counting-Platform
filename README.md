# Crowd-Counting-Platform
A platform that implements 4 crowd counting algorithms and helps you to predict number of people in images, videos, and in your camera. It would be used to facilitate crowd counting and face detection for the users.

**CrowdCounting.AI** include the latest and the most powerful crowd counting algorithms that use different crowd counting approaches. It combines both heavyweight models and lightweight models. The heavyweight models can be used both on server-based platforms and powerful devices to get the best accuracy when dealing with dense crowds, while lightweight
models can be used for real-time applications on devices with limited computational resources.


## Screenshots

### Home
![crowdcounting-website](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/ccb5a186-75de-4cbe-8bdf-a3d3d01618af)



### Prediction with FIDTM
#### Image
![FIDTM-Image](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/070d0928-26a2-46c0-93fd-674b4a5df607)
#### Video
![FIDTM-Video](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/f541cba7-5a73-4f38-ad23-af27239e2f1f)
### Prediction with P2PNet
#### Image
![P2PNet-Image](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/4032430f-0236-43a4-8f83-32c4fbf22a89)
#### Video
![FIDTM-Video](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/f541cba7-5a73-4f38-ad23-af27239e2f1f)

### Prediction with CSRNet
#### Image
![CSRNet-Image](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/73b6bfad-36ce-4355-9721-e63be6b6e6aa)
#### Video
![FIDTM-Image](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/070d0928-26a2-46c0-93fd-674b4a5df607)

### Prediction with YOLO-CROWD
#### Image
![Capture d’écran du 2023-06-13 20-17-55](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/5d81bad6-bf85-46e3-b026-e3bc3ee1f6ee)
#### Video
![screen_YOLO](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/41ae3078-7b21-454b-a6bc-14572d284dd8)


## Getting Started
1. Pull this repository
2. Install requirements: ` $ pip install -r requirements.txt `.
3. Download the models from the links bellow:
   - FIDTM: https://drive.google.com/file/d/1drjYZW7hp6bQI39u7ffPYwt4Kno9cLu8/view?usp=sharing
   - P2PNet: https://drive.google.com/file/d/1-189sscpNZBFaSHOz7dnEgAaFeUALiow/view?usp=sharing
   - CSRNet: https://drive.google.com/file/d/16PaKCz37FCe6ARsjFQK8lcQ6BeU-U3X1/view?usp=sharing
   - YOLO-CROWD:  https://drive.google.com/file/d/1xxXVCzseuzmHv7NoMQ03RVU_tDisWXjM/view?usp=sharing

    
5. Run the website: ` $ python app.py `.  

 

NB: If you want to use TensorRT you can download yolo-crowd.engine:
   https://drive.google.com/file/d/1-189sscpNZBFaSHOz7dnEgAaFeUALiow/view?usp=sharing



   


