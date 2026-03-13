import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from torch import cuda
import os
import json
import shutil
from tqdm import tqdm
import argparse
import sys
sys.path.append("Utils")
from SegmentUtil import segment
sys.path.append("Utils/Dataset-3DPOP")
from POP3D_Reader import Trial

def parse_args():
    parser = argparse.ArgumentParser(description='Create the training dataset from 3d-POP for MvP')
    parser.add_argument('--path', help='Path to 3D-POP dataset',
                        required=True, type=str)
    default_path=os.path.join(os.path.dirname(__file__),'pop3d')
    parser.add_argument('--out', default=default_path,
                        help='Path where training dataset is created')
    parser.add_argument('--seg', default=False,
                        help='Choose whether the dataset should be segmented')
    parser.add_argument('--sam', default="models/sam_vit_h_4b8939.pth",
                        help='Path to the SAM model weights (vit_h)')
    args, rest = parser.parse_known_args()
    return args
  

def createDataset(pop3d_path, out_path, sam_ckpt):
  """Creates a training dataset from 3D-POP for MvP in the format of the panoptic dataset. 
    The sampled N6000 folder of the 3D-POP dataset is used.

    Parameters
    ----------
    pop3d_path : str
        The Path to the 3D-POP dataset
    out_path : str
        The path where the dataset is created this path must not exist
    sam_ckpt : str
        The path to the segment anything (SAM) checkpoint(vit_h). If set to None, the images are not segmented
    Returns
    -------
    numpy.ndarray
        The segmented frame
    """
  data_path=os.path.join(pop3d_path,"N6000")
  keypoints=["hd_beak", "hd_leftEye", "hd_rightEye", "hd_nose", "bp_leftShoulder", "bp_rightShoulder", "bp_topKeel", "bp_bottomKeel", "bp_tail"]

  if not sam_ckpt==None:
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
    sam.to(device="cuda" if cuda.is_available() else "cpu")
    samPredictor = SamPredictor(sam)

  for typ in ["Train", "Val"]:
    print("Creating "+typ+" dataset...")
    anno_path=os.path.join(data_path,"Annotation", typ+"-3D.json")
    with open(anno_path, 'r') as file:
      annotation = json.load(file)
    annotation = annotation["Annotations"]
    for anno in tqdm(annotation):
      path=""     
      seq=""
      for cam in anno["CameraData"]:
        # copy image
        name=cam["CamName"]
        path=cam["Path"].split("/")[-1]
        seq=path.split("-")[0]
        out_dir=os.path.join(out_path,seq+"-"+typ,"Images",name)
        os.makedirs(out_dir, exist_ok=True)
        img_source=os.path.join(data_path,typ,name,path)
        img_dest=os.path.join(out_dir,path)    
        # segment if necessary
        if not sam_ckpt==None:
          bboxes=np.array(list(cam["BBox"].values()))
          frame=cv2.imread(img_source)
          frame=segment(samPredictor,frame,bboxes)
          cv2.imwrite(img_dest, frame)
        else:
          shutil.copy(img_source, img_dest)

      # create annotation file 
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
        
      # create camera calibration file
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
          T=np.array(cam.tvec)
          # Build extrinsic matrix [R | T]
          E = np.eye(4)
          E[:3, :3] = R
          E[:3, 3:] = T

          # Compute camera center in world coordinates: C = -R^{-1} T
          C = -np.linalg.inv(R) @ T

          # Homogeneous world point [-R^{-1}T, 1]
          Xw = np.vstack((C, [[1]]))

          # Transform to camera coordinates
          Xc = E @ Xw

          print("World point Xw:")
          print(Xw)

          print("\nTransformed point Xc:")
          print(Xc)

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
  sam_ckpt=None
  if args.seg:
    sam_ckpt=args.sam
  createDataset(args.path, args.out,sam_ckpt)

      
      
    
