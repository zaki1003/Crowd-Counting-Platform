import os

from flask import Flask, render_template, request, redirect, url_for, Response
from camera_fidtm import VideoCamera as VideoCameraFIDTM
from camera_p2pnet import VideoCamera as VideoCameraP2PNet
from camera_csrnet import VideoCamera as VideoCameraCSRNet



import cv2
from inferenceFIDTM import get_prediction as get_prediction_fidtm





from inferenceP2PNet import get_prediction as get_prediction_p2pnet


from threading import Thread, Event


from inferenceCRNet import get_prediction as get_prediction_crnet



from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = set(['.jpg', '.jpeg','mp4'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global event 
event = Event()


method = ''


@app.route('/', methods=['GET', 'POST'])
def upload_file():


    if request.method == 'POST':
        method = request.form['method']
        print('method',method)
        # Remove existing images in directory
        

        
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

            if( file_name1.endswith(".mp4") ): 
                return render_template('result-video.html',File=file_name1, Method=method) 

            if method == 'fidtm':
                prediction, density = get_prediction_fidtm(file_name1)
            elif method == 'p2pnet':
                prediction, density = get_prediction_p2pnet(file_name1)
            else:
                prediction, density = get_prediction_crnet(file_name1)

       
       
            if(  file_name1.endswith(".jpg") or file_name1.endswith(".jpeg") ):
                return render_template('result-image.html', Prediction=prediction, File=filename, Density=density , Method=method) 
            elif( file_name1.endswith(".mp4") ): 
                return render_template('result-video.html',File=filename, Method=method) 
        elif request.form.get('use-webcam') == 'use-webcam':  
            return render_template('result-video.html', File ='',Method=method) 
          
    return render_template('index.html')
    
 
    


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')






@app.route("/video_feed", methods = ['GET', 'POST'])
def video_feed():
    method = request.args.get('method', None)
    fileName = request.args.get('fileName', None)
    print("fileName", fileName)
    
    
    if method == 'fidtm':
        return Response(gen(VideoCameraFIDTM(fileName)),        
                mimetype='multipart/x-mixed-replace; boundary=frame') 
    elif method == 'p2pnet':
        return Response(gen(VideoCameraP2PNet(fileName)),        
                mimetype='multipart/x-mixed-replace; boundary=frame') 
    else:
        return Response(gen(VideoCameraCSRNet(fileName)),        
                mimetype='multipart/x-mixed-replace; boundary=frame') 
    
   

    
if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    app.run(threaded=True,host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
