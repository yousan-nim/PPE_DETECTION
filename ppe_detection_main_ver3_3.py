
import cv2
import numpy 
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from distutils.util import strtobool
from sort.sort import *
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
    
    mot_tracker = Sort() 
    
    video_path = f'extras/video/{video_path_param}'
    save_path  = f'extras/video_save/{save_path_param}'

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3)); frame_height =int(cap.get(4)) 
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    print("strat...")
    store_workers = [] 
    i_frame = 1
    number_pass_frame = 10

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
        worker_class, object_class = saparate_worker_class(image[0].numpy(), hat_status=True, vest_status=True) # return list

        #if worker_class !=[] :
        if worker_class.size > 0 :
            #print(numpy.array(worker_class))
            worker_class_track = mot_tracker.update(numpy.array(worker_class))
        else: 
            worker_class_track = None
        
        
        
        '''
        Draw boxes
        '''
        if not (worker_class_track is None):
            data_frames = PPE_DECISION(worker_class_track, 
				       object_class, 
				       class_names=class_names, 
                                       hat_status=True, 
				       vest_status=True, 
                                       i_frame=i_frame
					)
            store_workers.append(data_frames)

            if i_frame >= number_pass_frame + 2: 
                frame_average = window_frame(store_workers, back2n_frame=number_pass_frame)
                store_workers.pop(0)
                detected_img = PPE_DRAW_DETECTION(frame, 
						worker_class_track, 
						object_class, 
						class_names,
						enable, 
						visible=False, 
						#i_frame=i_frame, 
						frame_average=frame_average)

        else:
            detected_img = frame
        detected_img = frame
	
        print(i_frame)
        i_frame += 1
        
        '''
        Write VDO
        '''
        out.write(frame)
        
    print("finished...")



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
python ppe_detection_main_ver3_3.py -vdo full.mp4 -save test_test_test.avi -hardhat true -vest true -worker true -boots false -glove false 
python ppe_detection_main_ver3_3.py -vdo videoplayback.mp4 -save test_test_test.avi -hardhat true -vest true -worker true -boots false -glove false 
'''
