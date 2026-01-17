import argparse
import os
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO
from torch import cuda
import cv2
from tqdm import tqdm
import json
import numpy as np
import sys
sys.path.append("Utils")
from SegmentUtil import segment
sys.path.append("Utils/Dataset-3DPOP")
from POP3D_Reader import Trial




def parse_args():
    parser = argparse.ArgumentParser(description='Create the evaluation dataset from 3d-POP for MvP')
    parser.add_argument('--path', help='Path to 3D-POP dataset',
                        required=True, type=str)
    default_path=os.path.join(os.path.dirname(__file__),'pop3d-eval')
    parser.add_argument('--out', default=default_path,
                        help='Path where training dataset is created')
    parser.add_argument('--seg', default=False,
                        help='Choose whether the dataset should be segmented')
    parser.add_argument('--sam', default="models/sam_vit_h_4b8939.pth",
                        help='Path to the SAM model weights (vit_h)')
    parser.add_argument('--yolo', default="models/YOLO_Barn.pt",
                        help='Path to the YOLO model weights')
    args, rest = parser.parse_known_args()
    return args
 
  
def createSequence(SequenceNum,DatasetPath,out_path,startFrame,TotalFrames, samPredictor=None, YOLOModel=None):
    """Creates a dataset from a 3D-POP sequence for MvP in the format of the panoptic dataset. 

    Parameters
    ----------
    SequenceNum : int
        The number of the 3D-POP sequence to transform
    DatasetPath : str
        The Path to the 3D-POP dataset
    out_path : str
        The path where the dataset is created this path must not exist
    startFrame : int
        The number of the frame to start 
    TotalFrames : int
        The total number of the frames to extract
    samPredictor : segment_anything.predictor.SamPredictor
        The segment anything (SAM) predictor. If set to None, the images are not segmented
    YOLOModel : segment_anything.predictor.SamPredictor
        The YOLO-Model used for bounding box detection
    Returns
    -------
    numpy.ndarray
        The segmented frame
    """
    
    # load 3D-POP sequence
    SequenceObj = Trial.Trial(DatasetPath,SequenceNum)
    SequenceObj.load3DPopTrainingSet(Filter = True, Type = "Test")

    # prepare directories and store paths
    seq_out_path=os.path.join(out_path, SequenceObj.TrialName+"-Val")
    anno_dir=os.path.join(seq_out_path,'Annotation')
    os.makedirs(anno_dir)
    CamPaths = []
    for cam in SequenceObj.camObjects:
        cam_path=os.path.join(seq_out_path,'Images',cam.CamName)
        os.makedirs(cam_path)
        CamPaths.append(cam_path)

    # Setup video capture objects
    capList = []
    for cam in SequenceObj.camObjects:    
        cap = cv2.VideoCapture(cam.VideoPath)
        capList.append(cap)
    counter=startFrame    
    for cap in capList:
        cap.set(cv2.CAP_PROP_POS_FRAMES,counter) 
    if TotalFrames == -1:
        TotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    seq_name=SequenceObj.TrialName.split('_')[0]
    for i in tqdm(range(TotalFrames), desc = "Extracting "+seq_name):
        # extract frame
        for j,cap in enumerate(capList):
            ret, frame = cap.read()
            image_path=os.path.join(CamPaths[j],"%d.jpg" % counter)
            if not samPredictor==None:
                results = YOLOModel(frame, imgsz=3840, verbose = False)
                ##Filter for birds:
                classID = [key for key,val in results[0].names.items() if val == "bird"][0]
                DetectedClasses = results[0].boxes.cls.cpu().numpy().tolist()
                bboxes = results[0].boxes.xyxy.cpu().numpy().tolist()
                ##Filter birds only:
                bboxes = [box for x,box in enumerate(bboxes) if DetectedClasses[x] == classID]
                frame=segment(samPredictor,frame,bboxes)
            cv2.imwrite(image_path, frame)
        # create dummy annotation (else the datareader would not use the frame)       
        with open(os.path.join(anno_dir,"%d.json" % counter), "w") as fp:
            json.dump({"individuals": [{"id":"0","keypoints":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}, fp)
            
        if ret == False:
            break
        counter += 1
        
    # create camera calibration information 
    calib_path=os.path.join(seq_out_path,"calibration_"+SequenceObj.TrialName+"-Val.json")
    CamParamList=[]
    for cam in SequenceObj.camObjects:
      name=cam.CamName
      # convert camera calibration parameters
      R, _ = cv2.Rodrigues(cam.rvec)
      CamParamList.append({
                  "name": name,
                  "K":cam.camMat.tolist(),
                  "distCoef":cam.distCoef.tolist()[0],
                  "R":R.tolist(),
                  "t":cam.tvec.tolist()    
                  })
    CamDict={"cameras":CamParamList}
    with open(calib_path, "w") as fp:
      json.dump(CamDict , fp) 
    

if __name__ == "__main__":
    args = parse_args()
    evalSeqs=[1,2,5,11]
    samPredictor=None
    YOLOModel=None
    if args.seg:
        sam = sam_model_registry["vit_h"](checkpoint=args.sam)
        sam.to(device="cuda" if cuda.is_available() else "cpu")
        samPredictor = SamPredictor(sam)
        YOLOModel = YOLO(args.yolo)
    for SequenceNum in evalSeqs:
      createSequence(SequenceNum,args.path,args.out,startFrame=0,TotalFrames = 250,samPredictor=samPredictor, YOLOModel=YOLOModel)
    print("done.")    
    