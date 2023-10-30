import cv2
import numpy 
from time import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from sort.sort import *
from ppe_util_ver3_3 import *

from tensorflow.python.client import device_lib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.config.experimental.list_physical_devices('GPU')

# if len(physical_devices) > 0:
#     tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

class PPEDetect:
    def __init__(self, number_pass_frame):
        self.store_workers = []
        self.number_pass_frame = number_pass_frame
        self.mot_tracker = Sort() 
        self.input_shape, self.class_names, self.num_classes, self.anchor_boxes, self.model = prepare_yolo_model_object()


    def check_ppe_compliance_vdo(self, frame, num_frames, hat_status=True, vest_status=True, VDO=False): 
        compliance_flag = None 
        box_result = None

        # image detection 
        image = detection_frame(frame, self.input_shape, self.class_names, self.num_classes, self.anchor_boxes, self.model)
        # print("image: ",image)

        # worker tracking
        worker_class, object_classes = saparate_worker_class(image[0].numpy(), hat_status, vest_status)
        if worker_class.size > 0:
            worker_class_track = self.mot_tracker.update(worker_class)
        else: 
            worker_class_track = None
        
        # print("worker_class_track: ",worker_class_track)
        # decision
        if not (worker_class_track is None):
                data_frames = PPE_DECISION(worker_class_track, 
                                        object_classes, 
                                        hat_status, 
                                        vest_status, 
                                        self.class_names, 
                                        num_frames)
                self.store_workers.append(data_frames)
                # print(data_frames)
                # data frame return as (frame, id, true/false)
                # print("data_frames: ",data_frames)
                # if num_frames >= self.number_pass_frame + 5: 
                if num_frames >= self.number_pass_frame: 
                    frame_average = window_frame(self.store_workers, back2n_frame=self.number_pass_frame)
                    self.store_workers.pop(0)
                    __compliance_flage, __box_result = COMPLIANCE(frame_average, worker_class_track)
                    compliance_flag = __compliance_flage
                    box_result = __box_result
                    # draw box
                    if VDO:
                        detected_img = PPE_DRAW_DETECTION(frame, 
                                            worker_class_track, 
                                            object_classes, 
                                            self.class_names,
                                            visible=False, 
                                            i_frame=num_frames, 
                                            frame_average=frame_average)
                        if detected_img == []:
                            detected_im = frame 
                            
        return compliance_flag, box_result

    def compute_vdo(self, video_path, save_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3)); frame_height =int(cap.get(4)) 
        image_list = []
        frame_count = 1
        while True: 
            '''Get the image'''
            status, frame = cap.read() 
            if not status:
                break

            compliance_flag, box_result = self.check_ppe_compliance_vdo(frame, frame_count, VDO=True)
        
            frame_count += 1
            image_list.append(frame)
        self.save_vdo(image_list, save_path)

    def save_vdo(self, img_list, save_path):
        frame_height, frame_width, frame_ch = img_list[0].shape
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))
        for frame in img_list:
            out.write(frame) 
        out.release()


if __name__ == "__main__": 


    # video_path = 'extras/video/test2.mp4'
    # save_path  = './test_compliance.avi'
    # ppe =  PPEDetect(number_pass_frame = 15)
    # ppe.compute_vdo(video_path, save_path)


    # for single image feed into the function
    ppe =  PPEDetect(number_pass_frame = 1)
    img_path = "test_ppe.jpeg"
    img_path = "test_ppe2.jpg"
    # img_path = "test_ppe3.jpg"
    frame = cv2.imread(img_path)
    compliance_flag, box_result = ppe.check_ppe_compliance_vdo(frame, 1, VDO=False)

    # # for real-time feed image into the function
    # ppe =  PPEDetect(number_pass_frame = 15)
    # video_path = 'extras/video/test2.mp4'
    # cap = cv2.VideoCapture(video_path)
    # frame_width = int(cap.get(3)); frame_height =int(cap.get(4)) 
    # image_list = []
    # frame_count = 1
    # while True: 
    #     '''Get the image'''
    #     status, frame = cap.read() 
    #     if not status:
    #         break
    #     compliance_flag, box_result = ppe.check_ppe_compliance_vdo(frame, frame_count, VDO=False)
    #     if compliance_flag == True:
    #         # Sent warining
    #         pass 
    #     frame_count = frame_count + 1
    #     print("store_workers: ", ppe.store_workers)