import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import matplotlib as mpl

# from IPython.display import display, Math
from time import time

import sys
sys.path.append('../')

from src.utils.image import letterbox_image, draw_detection
from src.yolo3.model import yolo_body
from src.utils.fixes import *
# fix_tf_gpu()




def prepare_model(approach):
    '''
    Prepare the YOLO model
    '''
    global input_shape, class_names, anchor_boxes, num_classes, num_anchors, model

    # shape (height, width) of the imput image
    input_shape = (416, 416)

    # class names
    if approach == 1:
        class_names = ['H', 'V', 'W']

    elif approach == 2:
        class_names  = ['W','WH','WV','WHV']

    elif approach == 3:
        class_names  = ['W']

    else:
        raise NotImplementedError('Approach should be 1, 2, or 3')

    # anchor boxes
    if approach == 1:
        anchor_boxes = np.array(
            [
            np.array([[ 76,  59], [ 84, 136], [188, 225]]) /32, # output-1 anchor boxes
            np.array([[ 25,  15], [ 46,  29], [ 27,  56]]) /16, # output-2 anchor boxes
            np.array([[ 5,    3], [ 10,   8], [ 12,  26]]) /8   # output-3 anchor boxes
            ],
            dtype='float64'
        )
    else:
        anchor_boxes = np.array(
            [
            np.array([[ 73, 158], [128, 209], [224, 246]]) /32, # output-1 anchor boxes
            np.array([[ 32,  50], [ 40, 104], [ 76,  73]]) /16, # output-2 anchor boxes
            np.array([[ 6,   11], [ 11,  23], [ 19,  36]]) /8   # output-3 anchor boxes
            ],
            dtype='float64'
        )

    # number of classes and number of anchors
    num_classes = len(class_names)
    num_anchors = anchor_boxes.shape[0] * anchor_boxes.shape[1]

    # input and output
    input_tensor = Input( shape=(input_shape[0], input_shape[1], 3) ) # input
    num_out_filters = ( num_anchors//3 ) * ( 5 + num_classes )        # output

    # build the model
    model = yolo_body(input_tensor, num_out_filters)

    # load weights
    weight_path = f'./model-data/weights/pictor-ppe-v302-a{approach}-yolo-v3-weights.h5'
    model.load_weights( weight_path )

    
    
    
    
def detection(  prediction,
                anchor_boxes,
                num_classes,
                image_shape,
                input_shape,
                max_boxes = 20,
                score_threshold=0.3,
                iou_threshold=0.45,
                classes_can_overlap=True,):
    
    all_boxes  = []

    '''@ Each output layer'''
    for output, anchors in zip( prediction, anchor_boxes ):

        '''Preprocessing'''
        '''-------------'''
        # shapes
        batch_size     = output.shape[0]
        grid_h, grid_w = output.shape[1:3]

        # reshape to [batch_size, grid_height, grid_width, num_anchors, box_params]
        output = tf.reshape( output, [ -1, grid_h, grid_w, len(anchors), num_classes+5 ] )

        # create a tensor for the anchor boxes
        anchors_tensor = tf.constant(anchors, dtype=output.dtype)

        '''Scaling factors'''
        '''---------------'''
        image_shape_tensor = tf.cast( image_shape,       output.dtype ) # actual image's shape
        grids_shape_tensor = tf.cast( output.shape[1:3], output.dtype ) # grid_height, grid_width @ output layer
        input_shape_tensor = tf.cast( input_shape,       output.dtype )  # yolo input image's shape

        # reshape
        image_shape_tensor = tf.reshape( image_shape_tensor, [-1, 1, 1, 1, 2] )
        grids_shape_tensor = tf.reshape( grids_shape_tensor, [-1, 1, 1, 1, 2] )
        input_shape_tensor = tf.reshape( input_shape_tensor, [-1, 1, 1, 1, 2] )

        ### Scaling factors
        sized_shape_tensor = tf.round( image_shape_tensor * tf.reshape( tf.reduce_min( input_shape_tensor / image_shape_tensor, axis=-1 ), [-1,1,1,1,1] ) )
        # to scale the boxes from grid's unit to actual image's pixel unit
        box_scaling = input_shape_tensor * image_shape_tensor / sized_shape_tensor / grids_shape_tensor
        # to offset the boxes
        box_offsets = (tf.expand_dims(tf.reduce_max(image_shape_tensor, axis=-1), axis=-1) - image_shape_tensor) / 2.

        '''Box geometric properties'''
        '''------------------------'''
        grid_h, grid_w = output.shape[1:3] # grid_height, grid_width @ output layer

        grid_i = tf.reshape( np.arange(grid_h), [-1, 1, 1, 1] )
        grid_i = tf.tile( grid_i, [1, grid_w, 1, 1] )

        grid_j = tf.reshape( np.arange(grid_w), [1, -1, 1, 1] )
        grid_j = tf.tile( grid_j, [grid_h, 1, 1, 1] )

        grid_ji = tf.concat( [grid_j, grid_i], axis=-1 )
        grid_ji = tf.cast( grid_ji, output.dtype )

        # Box centers
        box_xy  = output[..., 0:2]
        box_xy  = tf.sigmoid( box_xy ) + grid_ji

        # Box sizes
        box_wh  = output[..., 2:4]
        box_wh  = tf.exp( box_wh ) * anchors_tensor

        # scale to actual pixel unit
        box_xy  = box_xy * box_scaling - box_offsets[...,::-1]
        box_wh  = box_wh * box_scaling

        # calculate top-left corner (x1, y1) and bottom-right corner (x2, y2) of the boxex
        box_x1_y1 = box_xy - box_wh / 2
        box_x2_y2 = box_xy + box_wh / 2

        # top-left corner cannot be negative
        box_x1_y1 = tf.maximum(0, box_x1_y1)
        # bottom-right corner cannot be more than actual image size
        box_x2_y2 = tf.minimum(box_x2_y2, image_shape_tensor[..., ::-1])

        '''Box labels and confidences'''
        '''--------------------------'''
        # class probabilities = objectness score * conditional class probabilities
        if classes_can_overlap:
            # use sigmoid for the conditional class probabilities
            classs_probs = tf.sigmoid( output[..., 4:5] ) * tf.sigmoid( output[..., 5:] )
        else:
            # use softmax for the conditional class probabilities
            classs_probs = tf.sigmoid( output[..., 4:5] ) * tf.nn.softmax( output[..., 5:] )

        box_cl = tf.argmax( classs_probs, axis=-1 )     # final classes
        box_sc = tf.reduce_max( classs_probs, axis=-1 ) # confidence scores

        '''Organize'''
        '''--------'''
        # take care of dtype and dimensions
        box_cl = tf.cast( box_cl, output.dtype )
        box_cl = tf.expand_dims(box_cl, axis=-1)
        box_sc = tf.expand_dims(box_sc, axis=-1)

        # store all information as: [ left(x1), top(y1), right(x2), bottom(y2),  confidence, label ]
        boxes  = tf.reshape( tf.concat( [ box_x1_y1, box_x2_y2, box_sc, box_cl ], axis=-1 ), 
                              [batch_size, -1, 6] )

        all_boxes. append( boxes  )

    # Merge across all output layers
    all_boxes  = tf.concat( all_boxes,  axis=1 )

    # To store all the final results of all images in the batch
    all_final_boxes = []

    '''For each image in the batch'''
    for _boxes_ in all_boxes:

        if classes_can_overlap:
            '''Perform NMS for each class individually'''

            # to stote the final results of this image
            final_boxes = []

            for class_id in range(num_classes):

                # Get the boxes and scores for this class
                class_boxes  = _boxes_[ _boxes_[...,-1] == class_id ]

                '''Non-max-suppression'''
                selected_idc = tf.image.non_max_suppression(
                    class_boxes[...,:4], # boxes' (y1,x1,y2,x2)
                    class_boxes[...,-2], # boxes' scores
                    max_output_size = max_boxes,
                    iou_threshold = iou_threshold,
                    score_threshold = score_threshold
                )

                # boxes selected by nms
                class_boxes = tf.gather( class_boxes,  selected_idc )
                final_boxes.append( class_boxes )

            # concatenate boxes for each class in the image
            final_boxes  = tf.concat( final_boxes,  axis=0 )

        else:
            '''Perform NMS for all classes'''

            # nms indices
            selected_idc = tf.image.non_max_suppression(
                _boxes_[...,:4], # boxes' (y1,x1,y2,x2)
                _boxes_[...,-2], # boxes' scores
                max_output_size = max_boxes,
                iou_threshold = iou_threshold,
                score_threshold = score_threshold
            )
            
            # boxes selected by nms
            final_boxes = tf.gather( _boxes_,  selected_idc )

        # append final boxes for each image in the batch
        all_final_boxes.append( final_boxes )
        
    return all_final_boxes



def get_detection(img):
    # save a copy of the img
    act_img = img.copy()

    # shape of the image
    ih, iw = act_img.shape[:2]

    # preprocess the image
    img = letterbox_image(img, input_shape)
    img = np.expand_dims(img, 0)
    image_data = np.array(img) / 255.

    # raw prediction from yolo model
    prediction = model.predict(image_data)

    # process the raw prediction to get the bounding boxes
    boxes = detection(
        prediction,
        anchor_boxes,
        num_classes,
        image_shape = (ih, iw),
        input_shape = (416,416),
        max_boxes = 10,
        score_threshold=0.3,
        iou_threshold=0.45,
        classes_can_overlap=False)

    # convert tensor to numpy
    boxes = boxes[0].numpy()

    # draw the detection on the actual image
    return draw_detection(act_img, boxes, class_names)