# Most code from (C) https://www.youtube.com/watch?v=Pb3opEFP94U

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

import cv2
import numpy as np
import os 
import subprocess
import shutil
from PIL import Image
import time
from pathlib import Path

from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

import matplotlib.pyplot as plt
from roboflow import Roboflow
import glob
import json
from threading import Thread

import fiftyone as fo
import fiftyone.zoo as foz

class Detector:
    def __init__(self, model_type = "OD", thresh_hold = 0.4):
        self.cfg = get_cfg()
        self.model_type = model_type

        if model_type == "OD": # object detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "IS": # instance segementions
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "KP": # keypoint detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "LVIS": 
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif model_type == "PS": # Panoptic Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        elif model_type == "XOD": # OD with X101-FPN
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")


        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_hold
        self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath, store=True):
        image = cv2.imread(imagePath)
        if self.model_type != "PS":
            predictions = self.predictor(image)

            viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                instance_mode = ColorMode.IMAGE)

            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        else:
            predicitions, segmentInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(predicitions.to("cpu"), segmentInfo)

        if store is True:
            new_path = os.path.join(os.path.abspath(os.path.dirname(imagePath)), os.pardir, "obj_det_img")
            file_name = os.path.basename(imagePath)
            file_name = file_name.split(".")[0]

            os.chdir(new_path)
            
            cv2.imwrite(f"{file_name}_detectron2.png", output.get_image()[:,:,::-1])    

        return predictions

def merge_image(img1_path, img2_path, save_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    img1_size = img1.size
    img2_size = img2.size

    new_img = Image.new('RGB', (2*img1_size[0], img2_size[1]), (250, 250, 250))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1_size[0], 0))

    new_img.save(save_path)


if __name__ == "__main__":
    
    # INFO: The way this currently works is really stupid and and should be changed
    # You should be able to use yolov7 as a package
    # TODO: FIX THIS TO BE DECENTLY CODED

        
    if False:

        dataset_dir = os.path.join(Path(__file__).parent, "valid_detectron2")
        labels_path = os.path.join(dataset_dir, "annotations.json")

        dataset_type = fo.types.COCODetectionDataset

        dataset = fo.Dataset.from_dir(
            data_path=dataset_dir,
            labels_path=labels_path,
            dataset_type=dataset_type,
        )

        session = fo.launch_app(dataset)
        session.wait()


    if True:

        cfg = get_cfg()
        
        #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

        #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.65
        cfg.MODEL.DEVICE = "cpu"

        predictor = DefaultPredictor(cfg)
        model = predictor.model

        output_path = os.path.join(Path(__file__).parent, "detectron2_output")
        dataset_path = os.path.join(Path(__file__).parent, "valid_detectron2")
        
        register_coco_instances("Kitchen-stuff", {}, f"{dataset_path}/annotations.json", dataset_path)


        evulator = COCOEvaluator(dataset_name="Kitchen-stuff", output_dir=output_path)
    
        test_loader = build_detection_test_loader(cfg, "Kitchen-stuff")

        inference_on_dataset(model, test_loader, evulator)


    if False:
        
        # Setting paths
        object_recogintion_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)
        yolov7_path = os.path.join(object_recogintion_path, "Bachelorarbeit_files", "yolov7")
        images_path = os.path.join(object_recogintion_path, "Bachelorarbeit", "Bilder", "docu_th_05_w6")
        obj_det_img_path = os.path.join(images_path, "obj_det_img")
        
        # User defined
        detectron_thr = 0.5
        yolov7_thr = str(detectron_thr)
        define_img = False

        kitchen_imgs_path = os.path.join(images_path, "og_img")
        file_list = [f for f in os.listdir(kitchen_imgs_path) if os.path.isfile(os.path.join(kitchen_imgs_path, f))]


        if define_img is True:
            # Define which images to use
            file_list = ["kitchen_image_20230918_1353_2.png"]
        else:
            # Use every image in images/og_img
            kitchen_imgs_path = os.path.join(images_path, "og_img")
            file_list = [f for f in os.listdir(kitchen_imgs_path) if os.path.isfile(os.path.join(kitchen_imgs_path, f))]
        
        for img_filename in file_list:

            kitchen_img_path = os.path.join(images_path, "og_img", img_filename)
            print(kitchen_img_path)
            # Starting detectron and yolov7
            command = ["python3", "detect.py", "--source", kitchen_img_path, "--project", images_path, \
                    "--name", "obj_det_img", "--exist-ok", "--conf-thres", yolov7_thr, "--save-txt", "--weights", "yolov7-w6.pt"]

            os.chdir(yolov7_path)
            proc = subprocess.Popen(command)

            #detector = Detector(model_type="XOD", thresh_hold=detectron_thr)
            #detector.onImage(kitchen_img_path, store=True)
            proc.wait()

            
            # Setting up new yolov7 file name
            source_path = os.path.join(obj_det_img_path, img_filename)
            new_img_filename = img_filename.split(".")[0]
            destination_path = os.path.join(obj_det_img_path, f"{new_img_filename}_yolov7.png")

            # Renaming the yolov7 .png
            try:
                shutil.move(source_path, destination_path)
            except:
                print("Error while moving the file.")
            
            #img_merging_path = os.path.join(obj_det_img_path, new_img_filename)
            #merge_image(f"{img_merging_path}_yolov7.png", f"{img_merging_path}_detectron2.png", 
            #    os.path.join(images_path, "combined_obj_det_img", f"{new_img_filename}_combined.png"))
            