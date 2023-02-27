from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import regularizers, optimizers
# import keras_tuner as kt
import statistics
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, array_to_img
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121

import numpy as np
import os
from skimage import io
from skimage import measure
import pydicom
from skimage.transform import resize
import cv2

import pickle



class Custom_Object_detection_inference:
    def __init__(self,classifier_weights_path,object_locolization_model_path):
        #loading the model architecture
        
        #loading classifier part..
        dense_net_121 = DenseNet121(input_shape=[224,224] + [3],include_top=False,pooling='avg')
        base_model_output = Dense(units=14,activation='relu')(dense_net_121.output)
        base_model = Model(inputs = dense_net_121.input,outputs=base_model_output)
#         base_model.load_weights(r'C:\Users\uUUUUvvvv\Desktop\projects\capstone_project\pre-trained_weights\brucechou1983_CheXNet_Keras_0.3.0_weights (1).h5')
        # loading pretrained weights 
        output_layer = Dense(1,activation='sigmoid')(base_model.layers[-2].output)
        self.chexnet_121_model = Model(inputs=base_model.inputs, outputs=output_layer)
        self.chexnet_121_model.load_weights(classifier_weights_path)
        print("chexnet_classifier loaded......")
        
        #loading the locolization model..
        # define iou or jaccard loss function
        def iou_loss(y_true, y_pred):
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])
            print(y_true)
            print(y_pred)
            # print(type(y_true * y_pred))
            intersection = tf.reduce_sum(y_true * y_pred)
            score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
            return 1 - score
        def mean_iou(y_true, y_pred):
            y_pred = tf.round(y_pred)
            intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
            union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
            smooth = tf.ones(tf.shape(intersect))
            return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))
        # combine bce loss and iou loss
        def iou_bce_loss(y_true, y_pred):
            print(y_true)
            print(y_pred)
            print(tensorflow.keras.losses.binary_crossentropy(y_true, y_pred))
            return 0.5 * tensorflow.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)
        
        
        self.locolization_model = tf.keras.models.load_model(object_locolization_model_path,custom_objects={'mean_iou':mean_iou,
                                                                     'iou_bce_loss':iou_bce_loss})
        print('Custom locolization model loaded....')
    
    def read_dicom(self,path_image):
        dicom_obj:pydicom.dataset.FileDataset = pydicom.read_file(path_image)
        pixels_values:np.array = dicom_obj.pixel_array
        return pixels_values
    
    def basic_pre_processing_for_classifier(self,pixels_values):
        # pixels_values = cv2.cvtColor(pixels_values, cv2.COLOR_GRAY2RGB)
        #creating extra dimension
        pixels_values:np.array = np.stack((pixels_values,) * 3, -1)
        #resizing the image
        pixels_values:np.array = cv2.resize(pixels_values, (224,224),interpolation = cv2.INTER_NEAREST).astype('float16')
        #scaling the image.
        pixels_values:np.array = pixels_values/pixels_values.max()
        return pixels_values
    
    def get_preprocess_image_for_classifier(self,dicom_image_path:str):
        image_array:np.array = self.read_dicom(dicom_image_path)
        pre_processed_image = self.basic_pre_processing_for_classifier(image_array)
        return pre_processed_image
    
    def predict_label(self,dicom_image_path):
        pre_processed_image = self.get_preprocess_image_for_classifier(dicom_image_path)
        pre_processed_image = np.expand_dims(pre_processed_image,axis = 0)
        prediction = self.chexnet_121_model.predict(pre_processed_image)
        print(prediction[0][0])
        if prediction[0][0] >= 0.5:
            return 'Pneumonia'
        else:
            return 'No Pneumonia'        
    
    def predict_bbox(self,dicom_image_path):
        image_array:np.array = self.read_dicom(dicom_image_path)
        image_array = resize(image_array, (128, 128), mode='symmetric')
        image_array = np.expand_dims(image_array, -1)
        image_array = np.expand_dims(image_array, 0)
        pred = self.locolization_model.predict(image_array)
        comp = pred[0,:, :, 0] > 0.8
        # apply connected components
        comp = measure.label(comp)
        # apply bounding boxes
        predictionString = ''
        pre_coordinates = []
        for region in measure.regionprops(comp):
            # retrieve x, y, height and width
            y, x, y2, x2 = region.bbox
            height = y2 - y
            width = x2 - x
            pre_coordinates.append((x,y,height,width))
        return image_array[0,:,:,0],pre_coordinates
              