
import cv2
import numpy 
import argparse
from time import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from distutils.util import strtobool
from ppe_util_ver3_3 import (prepare_yolo_model_object,
                      detection_frame,
                      draw_detection,
                      saparate_worker_class,
                      window_frame,
                      PPE_DRAW_DETECTION,
                      PPE_DECISION,)
import tensorflow as tf
import tensorflow.keras.backend as K

def ppe_detection(video_path_param, save_path_param, enable):  

    input_shape, class_names, num_classes, anchor_boxes, model = prepare_yolo_model_object()
    #Tracker = Tracking(class_list = ['H', 'V', 'W'])

    '''
    Writing vdo process
    '''
    video_path = f'extras/video/{video_path_param}'
    save_path  = f'extras/video_save/{video_path_param}'

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3)); frame_height =int(cap.get(4)) 
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    t0 = time() # set a timer
    print("strat...")
    print(enable, '**********************')

    Store_Worker = [] 

    '''While loop'''
    while True: 
        '''Get the image'''
        status, frame = cap.read() 
        if not status:
            break
            
            
            
        '''
        Detection
        '''
        image = detection_frame(frame, input_shape, class_names, num_classes, anchor_boxes, model)
        
        
        
        '''
        Tracking workers 
        '''
        worker_class, object_class = saparate_worker_class(image[0].numpy(), enable) # return list
        if worker_class:
            worker_class_track = Tracker.track_return_objs(frame, numpy.array(worker_class))
        else: 
            print("empty  numpy.array(worker_class) ")
        
        
        
        '''
        Draw boxes
        '''
        if worker_class_track:
            combine = combine_boxx(numpy.array(worker_class), numpy.array(worker_class_track))
            detected_img = draw_detection(frame, combine, object_class, class_names, enable, visible=True)
        else: 
            detected_img = frame 
        
        
        
        '''
        Write VDO
        '''
        out.write(frame)

    print('time taken to process : {:.2f} ms'.format( (time()-t0)*1000 ))


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("-vdo",  "--vdo",     type=str, help="VDO PATH")
    parser.add_argument("-save", "--save",    type=str, help="SAVE VDO PATH")
    parser.add_argument("-hardhat",    "--Hardhat", type=strtobool, default=True ,  help="enable hardhat default false")
    parser.add_argument("-vest",       "--Vest",    type=strtobool, default=True ,  help="enable vest default false")
    parser.add_argument("-worker",     "--Worker",  type=strtobool, default=True ,  help="enable worker default false")
    parser.add_argument("-boots",      "--Boots",   type=strtobool, default=False,  help="enable boots default false")
    parser.add_argument("-glove",      "--Glove",   type=strtobool, default=False,  help="enable glove default false")
    
    args = vars(parser.parse_args())

    # H, V, W, B, G, 
    Hardhat = True
    Vest    = True
    Worker  = True 
    Boots   = False
    Glove   = False
    
    if Hardhat == 0 : 
        Hardhat = False
    if Vest == 0 : 
        Vest    = False
    if Worker == 0 : 
        Worker  = False
    if Boots == 0 : 
        Boots   = False
    if Glove == 0 : 
        Glove   = False
    
    enable = [Hardhat, Vest, Worker, Boots, Glove]
    
    # Function detection here !
    K.clear_session()
    
    ppe_detection(video_path_param = args["vdo"], 
                  save_path_param  = args["save"], 
                  enable = enable) 
    
if __name__ == '__main__':
    main()

# video_path = 'extras/video/full.mp4'
# save_path  = 'extras/video_save/test_version3_True.avi'

'''
python ppe_detection_main.py -vdo full.mp4 -save test.avi -hardhat true -vest true -worker true -boots false -glove false 
'''
