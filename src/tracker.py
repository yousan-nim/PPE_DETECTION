from __future__ import division
from __future__ import print_function

import numpy as np

from src.yolo3.model import *
from src.yolo3.detect import *
from src.utils.image import *
from src.utils.datagen import *
from src.utils.fixes import *

from src.tracking.deep_sort import nn_matching
from src.tracking.deep_sort.detection import Detection 
from src.tracking.deep_sort.tracker import Tracker 
from src.tracking.tools import generate_detections as gdet

fix_tf_gpu()

class Tracking() :
    def __init__(self, weight_path='src/tracking/model_data/mars-small128.pb', 
                       nms_max_overlap = 0.8, 
                       max_cosine_distance = 0.5, 
                       nn_budget = None, 
                       class_list= ['W','WH','WV','WHV']) :
        
        self.weight_path = weight_path
        self.class_list = class_list
        self.nms_max_overlap = nms_max_overlap
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.setup()

    def setup(self) :
        self.encoder = gdet.create_box_encoder(self.weight_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric('cosine', self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(metric)

    def convert2deepsort_format(self, frame, boxes, xyhw=False):
#         print(boxes)
        boxes_dsort = boxes[:,:-2].copy()
#         print("boxes_dsort " , boxes_dsort)

        if xyhw :
            boxes_dsort[:, 2] = boxes_dsort[:, 2] - boxes_dsort[:, 0]
            boxes_dsort[:, 3] = boxes_dsort[:, 3] - boxes_dsort[:, 1]

        confidences = boxes[:,-2]
        class_arrs = boxes[:,-1]
        class_name_arrs = np.array([self.class_list[int(class_arrs[i])] for i in range(len(class_arrs))])

        features = self.encoder(frame, boxes_dsort)
        return (boxes_dsort, confidences, class_name_arrs, features)

    def track_return_objs(self, frame, boxes) :
        
#         print(boxes)

        infos = self.convert2deepsort_format(frame, boxes)
        boxes_dsort, confidences, class_name_arrs, features = infos[0], infos[1], infos[2], infos[3]
        infos_iter = zip(boxes_dsort.tolist(), confidences, class_name_arrs, features)

        detections = [Detection(box_dsort, confidence, class_name_arr, feature) for box_dsort, confidence, class_name_arr, feature in infos_iter]
        #indices = preprocessing.non_max_suppression(boxes_dsort, class_name_arrs, self.nms_max_overlap, confidences)
        #detections = [detections[i] for i in indices]
        self.tracker.predict()
        self.tracker.update(detections)
        objs = self.tracker.tracks

        # change format
        list_objs = []
        for obj in objs:
            if not obj.is_confirmed() or obj.time_since_update > 1 :
                continue
            class_id = obj.get_class()
            bbox = obj.to_tlbr()
            id_ = obj.track_id
            
            if class_id == 'Worker':
                data = [bbox, class_id, id_]
            else:
#                 data = [bbox, class_id, id_]
#                 data = bbox[0],bbox[1],bbox[2],bbox[3], class_id, id_
#                 data1 = boxes[0][0],boxes[0][1],boxes[0][2],boxes[0][3], class_id, id_
        
                data = (bbox[0]) , (bbox[1]) , (bbox[2]), (bbox[3]) , class_id, id_
            
            list_objs.append(list(data))

        return list_objs


        