# env\Scripts\activate.bat
import os
import sys
from flask import request, redirect, render_template, send_file, jsonify
from server.app import app
from utils.inference import Custom_Object_detection_inference
from utils.saved_models_path import CLASSIFIER_PATH,LOCOLIZATION_PATH
import shutil
import cv2

#loading the models
detector_obj = Custom_Object_detection_inference(CLASSIFIER_PATH,LOCOLIZATION_PATH)

@app.route("/")
def index():
    return redirect("/Pneumonia-Detection-input-file")


# starting page age.
@app.route("/Pneumonia-Detection-input-file", methods=["POST", "GET"])
def Base_page():
    return render_template("index.html")

# page3
@app.route(
    "/Pneumonia-Detection-input-file/Pneumonia-Detection-result", methods=["POST", "GET"]
)
def processing_old_users_request():
    if request.method == "POST":
        dicome_files =  request.files.getlist('files[]')
        print(dicome_files)
        #removing all files if exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        else:
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.makedirs(app.config['UPLOAD_FOLDER'])
        for file in dicome_files:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
        
        predicted_label = detector_obj.predict_label(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
        if not os.path.exists(os.path.join(os.getcwd(),'static','output_image')):
            os.makedirs(os.path.join(os.getcwd(),'static','output_image'))
        else:
            shutil.rmtree(os.path.join(os.getcwd(),'static','output_image'))
            os.makedirs(os.path.join(os.getcwd(),'static','output_image'))
        if predicted_label == 'Pneumonia':
            img,predicted_bbox = detector_obj.predict_bbox(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
            for region in predicted_bbox:
                # retrieve x, y, height and width
                y, x, y2, x2 = region
                height = y2 - y
                width = x2 - x
                print(f'The shape of the image :-  {img.shape}')
                cv2.rectangle(img,(x,y),(x+width,y+height),(0,0,255),1)
                # cv2.putText(img, 'Pneumonia', (x-40, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
            cv2.imwrite(os.path.join(os.getcwd(),'static','output_image','output.png'),img)
            return render_template("result_display.html",result = {'image_path':os.path.join(os.getcwd(),'static','output_image','output.png'),'label':'Pneumonia'})
        else:
            norma_image = detector_obj.read_dicom(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
            cv2.imwrite(os.path.join(os.getcwd(),'static','output_image','output.png'),norma_image)
            return render_template("result_display.html",result = {'image_path':os.path.join(os.getcwd(),'static','output_image','output.png'),'label':'No_Pneumonia'})


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
