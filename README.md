# Crowd-Counting-Platform
A platform that implements 4 crowd counting algorithms and helps you to predict the number of people in images, videos, and in your camera. It would be used to facilitate crowd counting and face detection for the users.

**CrowdCounting.AI** include the latest and the most powerful crowd counting algorithms that use different crowd counting approaches. It combines both heavyweight models and lightweight models. The heavyweight models can be used both on server-based platforms and powerful devices to get the best accuracy when dealing with dense crowds, while lightweight
models can be used for real-time applications on devices with limited computational resources.


## Screenshots

### Home
![crowdcounting-website](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/f3f1211a-18c5-4481-9601-bfe0aadadb2d)
### Prediction with FIDTM
#### Image
![FIDTM-Image](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/127eda64-dc7e-4cae-afde-31fedaf9830b)
#### Video
![FIDTM-Video](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/21764cce-c0f5-4a59-bb89-1e1d1acca841)
### Prediction with P2PNet
#### Image
![P2PNet-Image](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/f0329443-9d16-4c56-b28b-e326fb8c6582)
#### Video
![P2PNet-Video](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/4c2f6f6d-98e2-485f-8128-b2b9209f18c2)

### Prediction with CSRNet
#### Image
![CSRNet-Image](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/1390512a-bc3d-480b-baf9-2c157425fb75)
#### Video
![CSRNet-Video](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/43187f37-c35e-48b6-9adc-58ec404e5d70)


### Prediction with YOLO-CROWD
#### Image
![Capture d’écran du 2023-06-13 20-17-55](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/b9bbb1b1-2ddd-4ab1-9b0e-0583c2e4e4b0)
#### Video
![screen_YOLO](https://github.com/zaki1003/Crowd-Counting-Platform/assets/65148928/374ea941-2db9-42d3-9e3a-ecce032c2515)



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



   


