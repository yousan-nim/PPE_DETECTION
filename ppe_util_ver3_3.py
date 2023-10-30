import tensorflow as tf
import tensorflow.keras.backend as K


from tensorflow.keras.layers import Input

import numpy as np
import pandas as pd
import cv2

# from IPython.display import display, Math
from time import time
from collections import defaultdict

import sys
sys.path.append('../')

from src.utils.image import *
from src.yolo3.model import yolo_body
from src.utils.fixes import *
from src.detection import *
from sort.sort import *

# from src.tracker import Tracking

# fix_tf_gpu()
# K.clear_session()

global still_clear 
still_clear = []  

'''
#########################################################################################################
#                                                                                                       #
#                                           UTILLITY SUB FUNCTION                                       #
#                                                                                                       #
#########################################################################################################
'''

def saparate_worker_class(bbox, hat_status, vest_status):
    '''
    Saparate worker and objects 
    Enable/Unenable classs objects ['H','V','W',B','G']
    '''
    worker_status = True  # H, V, W, B, G, 
    
    class_worker = []
    class_object = []
    
    to_return = []
    
    for index, element in enumerate(bbox):
#             print(int(element[-1]))
            if hat_status:
                if int(bbox[index][-1]) == 0 : # H 
                    to_return.append(bbox[index])
                    
            if vest_status:
                if int(bbox[index][-1]) == 1 : # V 
                    to_return.append(bbox[index])
                    
            if worker_status:
                if int(bbox[index][-1]) == 2 & int(bbox[index][-1]) > .98 : # W  && int(bbox[index][-1]) > .95
                    to_return.append(bbox[index])
                    

    for i in range(len(to_return)):
        if int(to_return[i][-1]) == 2 : 
            class_worker.append(to_return[i]) # class : 2 worker
        else:
            class_object.append(to_return[i]) # class : 0 hat , 1 vest  
    
#     if class_worker:
#         mot_tracker.update(class_worker)
        
    return np.array(class_worker) , np.array(class_object)

def combine_boxx(x,y):
    combine_boxx = []
    for count, ele in enumerate(x):
        if count + 1 > len(y) :

            new = list(ele[:4]) + list(['W']) + list(['_'])
        else:
            new = list(ele[:4]) + list(y[count][4]) + list(y[count][5])
        combine_boxx.append(new)
    return combine_boxx

def saparate_clear(frame_average):
    clear = [] 
    not_pass = []
    for x in frame_average: 
        if frame_average[x][0] >= 0.3:
            clear.append(x)
        else:
            not_pass.append(x)
            if x in still_clear: 
                still_clear.remove(x)
    
    still_clear.extend(clear)

    return clear, not_pass






'''
#########################################################################################################
#                                                                                                       #
#                                           UTILLITY MAIN FUNCTIONs                                     #
#                                                                                                       #
#########################################################################################################
'''

def prepare_yolo_model_object(): 
    class_names = ['H', 'V', 'W']
    anchor_boxes = np.array([np.array([[ 76,  59], [ 84, 136], [188, 225]]) /32, 
                             np.array([[ 25,  15], [ 46,  29], [ 27,  56]]) /16, 
                             np.array([[ 5,    3], [ 10,   8], [ 12,  26]]) /8],dtype='float64')
    input_shape  = (416, 416)
    num_classes = len(class_names)
    num_anchors = anchor_boxes.shape[0] * anchor_boxes.shape[1]
    input_tensor = Input( shape=(input_shape[0], input_shape[1], 3) ) # input
    num_out_filters = ( num_anchors//3 ) * ( 5 + num_classes )        # output
    model = yolo_body(input_tensor, num_out_filters)
    weight_path = './model-data/weights/pictor-ppe-v302-a1-yolo-v3-weights.h5'
    model.load_weights(weight_path)

    return input_shape, class_names, num_classes, anchor_boxes, model


def window_frame(data, back2n_frame): 
    def mean(lst):
        return sum(lst)/3
    
    back_frame_summary = []
    
    data_i_frame = data[-1]
    sort_IDworker = [sort_id_worker[1] for sort_id_worker in data_i_frame]

    back2 = list(range(1, back2n_frame + 1))
    
    for n_frame in back2: 
        for ID in sort_IDworker:
            for back_frame in data[-n_frame]:
                if ID == back_frame[1] : 
                    export_back_frame = [ID, back_frame[2]]
                    back_frame_summary.append(export_back_frame)

    d = defaultdict(list)
    for id, *values in back_frame_summary:
        d[id].append(list(map(float, values)))

    out = {id: [mean(column) for column in zip(*values)] for id, values in d.items()}

    return out


def detection_frame(frame, input_shape, class_names, num_classes, anchor_boxes, model):
    image_data  = []
    image_shape = []
    all_boxes   = []
    
    max_boxes = 20,
    score_threshold=0.3,
    iou_threshold=0.45,
            
    classes_can_overlap=True
    
    image_shape.append(frame.shape[:-1] )
    image_data.append(letterbox_image(frame, (416,416))/255. )
    image_data  = np.array(image_data)
    image_shape = np.array(image_shape)
    
    prediction = model.predict(image_data, verbose=0)
    
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
            final_boxes = []
            for class_id in range(num_classes):
                class_boxes  = _boxes_[ _boxes_[...,-1] == class_id ]
                selected_idc = tf.image.non_max_suppression(
                    class_boxes[...,:4], # boxes' (y1,x1,y2,x2)
                    class_boxes[...,-2], # boxes' scores
                    max_output_size = 40,
                    iou_threshold = 0.45,
                    score_threshold = 0.4)
                class_boxes = tf.gather( class_boxes,  selected_idc )
                final_boxes.append( class_boxes )
            final_boxes  = tf.concat( final_boxes,  axis=0 )
        else:
            selected_idc = tf.image.non_max_suppression(
                _boxes_[...,:4], # boxes' (y1,x1,y2,x2)
                _boxes_[...,-2], # boxes' scores
                max_output_size = 40,
                iou_threshold = 0.45,
                score_threshold = 0.)
            final_boxes = tf.gather( _boxes_,  selected_idc )
        all_final_boxes.append( final_boxes )
        
    return all_final_boxes

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = abs(max((abs(xB - xA), 0)) * max(abs(yB - yA), 0))
    if interArea == 0:
        return 0

    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    
    iou =  interArea / boxBArea

    if iou <= 0:
        return 0
    if iou >=1 :
        return 1.0
    return iou

def PPE_DECISION(boxes, box_object, hat_status, vest_status, class_names, i_frame=None, visible=True):
    
    class_names = []
    store_worker = []
    
    if hat_status == True: 
        class_names.extend('H')
    if vest_status == True: 
        class_names.extend('V')
#     if enable[3] == True: 
#         class_names.extend('B')
#     if enable[4] == True: 
#         class_names.extend('G')
    
    for i in boxes: 
        '''
        Bonding box decision
        '''
        x1, y1, x2, y2 = int(float(i[0])), int(float(i[1])), int(float(i[2])), int(float(i[3]))
        not_in = [] 
        summary = [] 
        id_worker = i[-1]
        
        for j in box_object: 
            label = int(j[-1])
            if j[0] >= i[0] - 50 and j[2] <= i[2] + 50: 
                percentage = bb_intersection_over_union(i, j)
                if percentage >= 0.2 : 
                    summary.append(class_names[label])

        for k in class_names:
            if k not in summary: 
                not_in.append(k)
        
        if not_in == []:
            worker_store = [i_frame, int(id_worker), 1]
            store_worker.append(worker_store)
        else:
            worker_store = [i_frame, int(id_worker), 0]
            store_worker.append(worker_store)

    return store_worker
        
    
    
    
def PPE_DRAW_DETECTION(img,  
                       boxes,  
                       box_object,  
                       class_names,   
                       i_frame,
                       frame_average,
                       visible=False,
                       #TEXT PARAMETER
                       
                       font=cv2.FONT_HERSHEY_DUPLEX, 
                       font_scale=1, 
                       box_thickness=2, 
                       border=5, 
                       text_color=(255, 255, 255), 
                       text_weight=1,):
    
    clear, not_pass = saparate_clear(frame_average) 
    
#     print(not_pass)

#     print(list(set(still_clear)))
    
    still = list(set(still_clear))
    
    for i in boxes: 
        x1, y1, x2, y2 = int(float(i[0])), int(float(i[1])), int(float(i[2])), int(float(i[3]))
    
        if int(i[-1]) in clear or int(i[-1]) in still:
            clr = [0,255,0] # GREEN
            img = cv2.rectangle(img, (x1, y1), (x2, y2), clr, box_thickness)
            text = f"WOEKER({int(i[-1])}) CLEAR "
        elif int(i[-1]) in not_pass:
            clr = [0,0,255] # RED
            img = cv2.rectangle(img, (x1, y1), (x2, y2), clr, box_thickness)
#             text = f"WORKER({int(i[-1])}) NOT HAVE {str(not_in)[1:-1]}"
            text = f"WORKER({int(i[-1])}) NOT PASS "
        else: 
            clr = [45, 255, 255] # YELLOW
            img = cv2.rectangle(img, (x1, y1), (x2, y2), clr, box_thickness)
            text = f"PROCESSING"
        
        if int(i[-1]) in not_pass : 
            compliance_flag = int(i[-1])
            box_result = [x1, y1, x2, y2]
#             print(compliance_flag)
#             print(box_result)


        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX,0.5, 1)
        tb_x1 = x1 - box_thickness//2
        tb_y1 = y1 - box_thickness//2 - th - 2*border
        tb_x2 = x1 + tw + 2*border
        tb_y2 = y1
        img = cv2.rectangle(img, (tb_x1, tb_y1), (tb_x2, tb_y2), clr, -1)
        img = cv2.putText(img, text, (x1 + border, y1 - border), cv2.FONT_HERSHEY_DUPLEX,   0.5, text_color, text_weight, cv2.LINE_AA)
        img = cv2.putText(img, 'FRAME'+ str(i_frame) , (10, 30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, text_weight, cv2.LINE_AA)


    '''
    Draw boxs for objects 
    '''
    if visible == True:
        for box in box_object: 
            x1, y1, x2, y2 = int(float(box[0])), int(float(box[1])), int(float(box[2])), int(float(box[3]))
            score = box[-2]
            label = int(box[-1])
            clr = [255,0,0]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), clr, box_thickness)
            text = f'{class_names[label]}'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX,0.5, 1)
            tb_x1 = x1 - box_thickness//2
            tb_y1 = y1 - box_thickness//2 - th - 2*border
            tb_x2 = x1 + tw + 2*border
            tb_y2 = y1
            img = cv2.rectangle(img, (tb_x1, tb_y1), (tb_x2, tb_y2), clr, -1)
            img = cv2.putText(img, text, (x1 + border, y1 - border), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, text_weight, cv2.LINE_AA)
            
    return img
    
    
        
def draw_detection_verseion3(
    img,
    boxes,
    box_object,
    class_names,
    enable,
    visible=True,
    font=cv2.FONT_HERSHEY_DUPLEX,
    font_scale=1,
    box_thickness=2,
    border=5,
    text_color=(255, 255, 255),
    text_weight=1
):
    '''
    Draw the bounding boxes on the image
    '''
    num_classes = len(class_names) # number of classes
    colors = [mpl.colors.hsv_to_rgb((i/num_classes, 1, 1)) * 255 for i in range(num_classes)]

    '''
    Draw boxs for objects 
    '''
    if visible == True:
        for box in box_object: 
            x1, y1, x2, y2 = int(float(box[0])), int(float(box[1])), int(float(box[2])), int(float(box[3]))
            score = box[-2]
            label = int(box[-1])
            clr = [255,0,0]

            img = cv2.rectangle(img, (x1, y1), (x2, y2), clr, box_thickness)
            text = f'{class_names[label]}'

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX,0.5, 1)

            tb_x1 = x1 - box_thickness//2
            tb_y1 = y1 - box_thickness//2 - th - 2*border
            tb_x2 = x1 + tw + 2*border
            tb_y2 = y1

            img = cv2.rectangle(img, (tb_x1, tb_y1), (tb_x2, tb_y2), clr, -1)
            img = cv2.putText(img, text, (x1 + border, y1 - border), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, text_weight, cv2.LINE_AA)

    return img

def draw_detection_only_workers( # Use for draw only worker box 
    img,
    boxes,
    class_names,
    font=cv2.FONT_HERSHEY_DUPLEX,
    font_scale=0.5,
    box_thickness=1,
    border=5,
    text_color=(255, 255, 255),
    text_weight=1,
):
    '''
    Draw the bounding boxes on the image
    '''
    num_classes = len(class_names) 
    colors = [mpl.colors.hsv_to_rgb((i/num_classes, 1, 1)) * 255 for i in range(num_classes)]
    for box in boxes:
        x1, y1, x2, y2 = int(float(box[0])), int(float(box[1])), int(float(box[2])), int(float(box[3]))
        score = box[-2]
        label = int(box[-1])
        clr = [0,0,255]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), clr, box_thickness)
        text = f'{box[-1]}' 

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)

        tb_x1 = x1 - box_thickness//2
        tb_y1 = y1 - box_thickness//2 - th - 2*border
        tb_x2 = x1 + tw + 2*border
        tb_y2 = y1

        img = cv2.rectangle(img, (tb_x1, tb_y1), (tb_x2, tb_y2), clr, -1)
        img = cv2.putText(img, text, (x1 + border, y1 - border), font, font_scale, text_color, text_weight, cv2.LINE_AA)

    return img


def COMPLIANCE(frame_average, worker_class_track):
    clear, not_pass = saparate_clear(frame_average) 
    compliance_flage = False
    box_result = None
    for index , ele in enumerate(worker_class_track): 
        if ele[-1] in not_pass: 
#             print(worker_class_track[index][:3], not_pass)
            compliance_flage = True
            box_result = worker_class_track[index][:4]
        
        # else: 
        #     compliance_flage = False
        #     box_result = None

    return compliance_flage, box_result



def get_detection_(img, input_shape, class_names, num_classes, anchor_boxes, model):
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


def draw_detection_(
    img,
    boxes,
    class_names,
    # drawing configs
    font=cv2.FONT_HERSHEY_DUPLEX,
    font_scale=0.5,
    box_thickness=2,
    border=5,
    text_color=(255, 255, 255),
    text_weight=1
):
    '''
    Draw the bounding boxes on the image
    '''
    # generate some colors for different classes
    num_classes = len(class_names) # number of classes
    colors = [mpl.colors.hsv_to_rgb((i/num_classes, 1, 1)) * 255 for i in range(num_classes)]
    
    # draw the detections
    for box in boxes:
        print(box)
        x1, y1, x2, y2 = box[:4]
        score = box[-2]
        label = int(box[-1])

        clr = colors[label]

        # draw the bounding box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), clr, box_thickness)

        # text: <object class> (<confidence score in percent>%)
        text = f'{class_names[label]} ({score*100:.0f}%)'

        # get width (tw) and height (th) of the text
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)

        # background rectangle for the text
        tb_x1 = x1 - box_thickness//2
        tb_y1 = y1 - box_thickness//2 - th - 2*border
        tb_x2 = x1 + tw + 2*border
        tb_y2 = y1

        # draw the background rectangle
        img = cv2.rectangle(img, (tb_x1, tb_y1), (tb_x2, tb_y2), clr, -1)

        # put the text
        img = cv2.putText(img, text, (x1 + border, y1 - border), font, font_scale, text_color, text_weight, cv2.LINE_AA)

    return img

