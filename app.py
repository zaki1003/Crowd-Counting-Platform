import os

from flask import Flask, render_template, request, redirect, url_for, Response
from camera_csrnet import VideoCamera
import cv2
from inferenceFIDTM import get_prediction as get_prediction_fidtm
from inferenceFIDTM import get_prediction_webcam as get_prediction_fidtm_webcam




from inferenceP2PNet import get_prediction as get_prediction_p2pnet
from inferenceP2PNet import get_prediction_webcam as get_prediction_p2pnet_webcam

from threading import Thread, Event


from inferenceCRNet import get_prediction as get_prediction_crnet
from inferenceCRNet import get_prediction_webcam as get_prediction_csrnet_webcam


from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = set(['.jpg', '.jpeg','mp4'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global event 
event = Event()





@app.route('/', methods=['GET', 'POST'])
def upload_file():

    thread_fidtm_webcam = Thread(target = get_prediction_fidtm_webcam ,args=( event,))
    thread_p2pnet_webcam = Thread(target = get_prediction_p2pnet_webcam ,args=( event,))
    thread_csrnet_webcam = Thread(target = get_prediction_csrnet_webcam ,args=( event,))

    
    if request.method == 'POST':
        method = request.form['method']
        print('method',method)
        # Remove existing images in directory
        
        
        if request.form.get('stop-video') == 'stop-video':
            print("stop video")

            event.set()  
            print("exent_isSet: ",                 event.is_set())
            

            
           # event = Event()

        
        if request.form.get('upload-file') == 'upload-file':
            files_in_dir = os.listdir(app.config['UPLOAD_FOLDER'])
            filtered_files = [file for file in files_in_dir if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".mp4") ]
            for file in filtered_files:
                path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                os.remove(path)

            # Upload new file
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']

            if not file:
                return
            
            filename = secure_filename(file.filename)
            file_name1 =os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("GETTING PREDICTION")

            if method == 'fidtm':
                prediction, density = get_prediction_fidtm(file_name1)
            elif method == 'p2pnet':
                prediction, density = get_prediction_p2pnet(file_name1)
            else:
                prediction, density = get_prediction_crnet(file_name1)
            #prediction, density = get_prediction(file_name1)
            if(  file_name1.endswith(".jpg") or file_name1.endswith(".jpeg") ):
                return render_template('result.html', Prediction=prediction, File=filename, Density=density , Method=method) 
            
        elif request.form.get('use-webcam') == 'use-webcam':  
            
             return render_template('result-video.html', Method=method) 

                
    #        if method == 'fidtm':
    #            event.clear()  
    #            thread_fidtm_webcam.start()
     #       elif method == 'p2pnet':
      #          event.clear()  
       #         thread_p2pnet_webcam.start()
        #    else:
         #       event.clear()  
          #      thread_csrnet_webcam.start()

    
           
      #      return render_template('result-webcam.html', Method=method) 

            
    return render_template('index.html')
    
 
    


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame') 
 

    
if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    app.run(threaded=True,host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
