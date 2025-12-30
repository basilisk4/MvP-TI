import cv2 
import pandas as pd
import subprocess
import os
import json
import shutil
from tqdm import tqdm
import argparse
import sys
sys.path.append("repositories/Dataset-3DPOP")
from POP3D_Reader import Trial

def parse_args():
    parser = argparse.ArgumentParser(description='Create the training dataset from 3d-POP for MvP')
    parser.add_argument('--path', help='Path to 3D-POP dataset',
                        required=True, type=str)
    default_path=os.path.join(os.path.dirname(__file__),'pop3d')
    parser.add_argument('--out', default=default_path,
                        help='Path where training dataset is created')
    args, rest = parser.parse_known_args()
    return args

def main(pop3d_path, out_path):
  data_path=os.path.join(pop3d_path,"N6000")
  keypoints=["hd_beak", "hd_leftEye", "hd_rightEye", "hd_nose", "bp_leftShoulder", "bp_rightShoulder", "bp_topKeel", "bp_bottomKeel", "bp_tail"]

  for typ in ["Train", "Test"]:
    print("Creating "+typ+" dataset...")
    anno_path=os.path.join(data_path,"Annotation", typ+"-3D.json")
    with open(anno_path, 'r') as file:
      annotation = json.load(file)
    annotation = annotation["Annotations"]
    for anno in tqdm(annotation):
      path=""     
      seq=""
      for cam in anno["CameraData"]:
        name=cam["CamName"]
        path=cam["Path"].split("/")[-1]
        seq=path.split("-")[0]
        out_dir=os.path.join(out_path,seq+"-"+typ,"Images",name)
        os.makedirs(out_dir, exist_ok=True)
        img_source=os.path.join(data_path,typ,name,path)
        img_dest=os.path.join(out_dir,path)    
        shutil.copy(img_source, img_dest)

      frameList=[]
      gt_keypoints=anno["Keypoint3D"]
      for ind in anno["BirdID"]:
        indDict={"id": ind}
        keyList=[]
        ind_keypoints=gt_keypoints[ind]
        for key in keypoints:
          keyList+=ind_keypoints[key]
        indDict["keypoints"]=keyList
        frameList.append(indDict)
      frameDict={"individuals":frameList}
      anno_dir=os.path.join(out_path,seq+"-"+typ,"Annotation")
      os.makedirs(anno_dir, exist_ok=True)
      json_name=path.split(".")[0]+".json"
      json_path=os.path.join(anno_dir,json_name)
      with open(json_path, "w") as fp:
        json.dump(frameDict , fp)

      calib_path=os.path.join(out_path, seq+"-"+typ,"calibration_"+seq+"-"+typ+".json")
      if not os.path.isfile(calib_path):
        SequenceNum=seq.split("Sequence")[1].split("_")[0]
        SequenceObj = Trial.Trial(pop3d_path,SequenceNum)
        SequenceObj.load3DPopDataset()
        cams = SequenceObj.camObjects
        CamParamList = []
        for cam in cams:
          name=cam.CamName
          #Convert camera calibration parameters
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
  print("done.")


if __name__ == '__main__':
  args = parse_args()
  main(args.path, args.out)

      
      
    
