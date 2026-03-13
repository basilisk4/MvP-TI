import argparse
import pickle
import os 
import numpy as np
import csv
import math
import cv2
import json
from scipy.spatial.distance import pdist, squareform
from KalmanPostProcess import apply_kalman
from SeqEvaluationMetrics import runEvaluation
import sys
sys.path.append("Utils")
import VisualizeUtil
from tqdm import tqdm
import subprocess

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, 
                        help='experiment configure file name')
    parser.add_argument('--data', required=True ,type=str,
                        help='Path to dataset to do inference on')
    parser.add_argument("--name", type=str, required=True,
                        help="Name for the whole framework")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the 3DPOP dataset")

    arg = parser.parse_args()

    return arg

def VizualizeAll(path,Point3DDict, Lines = True):
    """Visualize all on visualize cam"""
    path=path.replace('pop3d-eval-seg','pop3d-eval')
    frame=cv2.imread(path)
    path=path.split('/')
    Cam=path[-2]

    calib_dir=os.path.join("/",*path[:-3])
    files = os.listdir(calib_dir)
    # Filter for JSON files
    json_files = [file for file in files if file.endswith('.json')]
    calib_path = os.path.join(calib_dir, json_files[0])
    with open(calib_path) as cfile:
            calib_data = json.load(cfile)
    VisCam={}
    for cam in calib_data['cameras']:
        if cam['name'] == Cam:
            VisCam['K'] = np.array(cam['K'])
            VisCam['distCoef'] = np.array(cam['distCoef'])
            VisCam['R'] = np.array(cam['R'])
            VisCam['t'] = np.array(cam['t']).reshape((3, 1))
    
    imsize=frame.shape[0], frame.shape[1]
    for Subject in Point3DDict:
        PointsDict = {}
        Points3DArr = np.array(Subject[:, :3])
        PointsNames = list(["hd_beak", "hd_leftEye", "hd_rightEye", "hd_nose", "bp_leftShoulder", "bp_rightShoulder", "bp_topKeel", "bp_bottomKeel", "bp_tail"])
        # import ipdb;ipdb.set_trace()
        try:
            Allimgpts, jac = cv2.projectPoints(Points3DArr, VisCam['R'], VisCam["t"], VisCam['K'], VisCam['distCoef'])
        except Exception as e: 
            print(e)
            continue

        for i in range(len(Allimgpts)):
            pts = Allimgpts[i]
            if np.isnan(pts[0][0]) or math.isinf(pts[0][0]) or math.isinf(pts[0][1]):
                continue
            #######
            point = (round(pts[0][0]),round(pts[0][1]))
            if VisualizeUtil.IsPointValid(imsize,point):
                colour = VisualizeUtil.getColor(PointsNames[i])
                cv2.circle(frame,point,1,colour, -1)
            
            PointsDict.update({PointsNames[i]:point})
    
        ##Plot Lines:
        if Lines:
            VisualizeUtil.PlotLine(PointsDict,"leftEye","nose",[255,0,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"rightEye","nose",[255,0,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"beak","nose",[255,0,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"leftEye","rightEye",[255,0,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"leftShoulder","rightShoulder",[0,255,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"leftShoulder","topKeel",[0,255,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"topKeel","rightShoulder",[0,255,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"leftShoulder","tail",[0,255,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"tail","rightShoulder",[0,255,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"tail","bottomKeel",[0,255,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"bottomKeel","topKeel",[0,255,0],frame)
                        
    return frame,VisCam

def VisualizeID(path,id_dict,VisCam):
    """Visualize all on visualize cam"""
    frame=cv2.imread(path)
    try:
        points2d,_=cv2.projectPoints(np.array([list(id_dict.values())]), VisCam['R'], VisCam["t"], VisCam['K'], VisCam['distCoef'])
    except:
        return
    for i,id in enumerate(id_dict):
        Point=points2d[i][0]
        Point=Point.ravel().astype(int)
        x,y = Point
        text = str(int(id))
        font = cv2.FONT_HERSHEY_COMPLEX
        text_size, _ = cv2.getTextSize(text, font, 1, 1)
        text_w, text_h = text_size
        cv2.rectangle(frame, (x-text_w, y-text_h), (x+text_w, y+text_h), [255,255,255], -1)
        frame = cv2.putText(frame,text,(x-text_w,y+text_h), font, 2, [255,0,0],2, cv2.LINE_AA)
    cv2.imwrite(path,frame)
                        

def select_preds(preds, n):
    #filter predictions tht are bigger than 30cm
    filtered_preds=[]
    for pred in preds:
        coordinates=pred[:,:3]
        distances=pdist(coordinates)
        size=np.max(distances)
        #print(size)
        if size<250:
            filtered_preds.append(pred)
    #return filtered_preds
    #select the predictions that are most likly therby only selectiing one for each bird        
    sorted_preds= sorted(filtered_preds, key=lambda x: x[0][4], reverse=True)
    sorted_preds=[pred[:,:3] for pred in sorted_preds]
    selected_preds=[]
    while len(selected_preds)<n and sorted_preds!=[]:
        head=sorted_preds[0]
        selected_preds.append(head)
        sorted_preds.pop(0)
        tail=sorted_preds
        sorted_preds=[]
        for pred in tail:
            distances = np.linalg.norm(head - pred, axis=1)
            if np.average(distances)>150.0:
                sorted_preds.append(pred)

    return selected_preds


def match_preds(frame_dict, n):
    matched_dict={}
    last_known={i:[] for i in range(n)}
    last_entry={i:[] for i in range(n)}
    for frame, preds in sorted(frame_dict.items()):
        #print(frame)
        entry={i:[] for i in range(n)}
        # match with detections of last frame
        for id, last_pred in last_entry.items():
            if len(last_pred)>0:
                match, preds,distance=match_one_pred(last_pred, preds,30)
                
                if len(match)>0:
                    #print(distance)
                    #print("Matched last entry "+str(id))
                    entry[id]=match
                    last_known[id]=match
            
        # try to match with later detection
        for id, pred in entry.items():
            if len(pred)==0 and len(last_known[id])>0:
                match, preds,distance=match_one_pred(last_known[id], preds,60)              
                if len(match)>0:
                    #print(distance)
                    #print("Matched last known entry "+str(id))
                    entry[id]=match
                    last_known[id]=match
                    
       # force match
        count_empty_ids = sum(1 for x in last_known.values() if len(x) == 0)
        force_count = max(0, len(preds) - count_empty_ids)
        min_distance =-1
        for _ in range(force_count):
            min_distance = float('inf')
            match_id = -1
            match = None
            match_preds = preds
            
            for id, com_entry in last_known.items():
                if len(entry[id]) == 0 and len(com_entry) > 0:
                    new_match, new_preds, distance = match_one_pred(com_entry, preds)
                    if distance < min_distance:
                        match = new_match
                        match_preds = new_preds
                        match_id = id
                        min_distance = distance
            
            if match_id != -1:
                
                entry[match_id] = match
                last_known[match_id] = match  # Update the last known position

            preds = match_preds  # Update preds for the next iteration
        #print(min_distance)       

            
        #fill empty ids
        for id, known_pred in last_known.items():
            if len(known_pred)==0 and len(preds)>0: 
                match=preds[0]
                preds.pop(0)
                #print("Matched empty entry")
                entry[id]=match
                last_known[id]=match   
        
        last_entry=entry
        matched_dict[frame]=entry
    return matched_dict
    

def match_one_pred(pred, pred_list, threshold=0):
    match = []
    min_distance = float('inf')
    match_index = -1

    for i, compare_pred in enumerate(pred_list):
        distance = np.average(np.linalg.norm(pred - compare_pred, axis=1))  
        if distance < min_distance and (distance < threshold or threshold == 0):
            match_index = i
            min_distance = distance

    if match_index != -1:
        match = pred_list[match_index]
        new_pred_list = [p for p in pred_list if not np.array_equal(p, match)]
    else:
        new_pred_list = pred_list

    return match, new_pred_list, min_distance



def export_results(results,SequenceNum,ModelName,path,VisCam):
    
    PointNames = ["hd_beak", "hd_leftEye", "hd_rightEye", "hd_nose", "bp_leftShoulder", "bp_rightShoulder", "bp_topKeel", "bp_bottomKeel", "bp_tail"]
    points3d={}
    tracker3d={}
    for frame, frame_results in tqdm(results.items()):
        framedict={}
        tracker={}
        for id, id_results in frame_results.items():
            for i in range(len(PointNames)):
                key=str(id)+'_'+PointNames[i]
                try:
                    framedict.update({key:(id_results[i].tolist())})
                    if PointNames[i]=="bp_bottomKeel":
                        tracker.update({str(id):list(id_results[i].tolist())})
                except Exception as e: 
                    #print(e)
                    framedict.update({key:float('nan')})
        image_path=os.path.join(path,'Images',f"{SequenceNum}-{frame}.jpg")
        VisualizeID(image_path,tracker,VisCam)
                
        points3d[int(frame)]=framedict
        tracker3d[int(frame)]=tracker
        
    pickle.dump(points3d,open(os.path.join(path,"SeqEval_Points3D_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))
    pickle.dump(tracker3d,open(os.path.join(path,"SeqEval_3DTracker_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))
    
    filename = os.path.join(path,"%s-%spoints3d.csv"%(ModelName,SequenceNum))
    # Writing to CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(next(iter(points3d.values())).keys())
        # Write the data rows
        for row in points3d.values():
            writer.writerow(row.values())
    




if __name__ == "__main__":
    args = ParseArgs()
    ModelName=args.name
    DatasetPath = args.dataset
    
    ParentDir=os.path.dirname(os.path.realpath(__file__))
    OutDir=os.path.join(ParentDir, ModelName)
    evalSeqs=[1,2,5,11]
    
    ind_dict={"1":1, "2":2,"5":5,"11":10}
    seq_dict={}
    
    ModelName='MvP'
    image_path=os.path.join(OutDir,'Images')
    os.makedirs(image_path)
    
    result_path=os.path.join(OutDir,'results.p')
    cmd = [
    "python",
    "-m", "torch.distributed.launch",
    "--nproc_per_node=1",
    "run/inference.py",
    "--cfg", args.cfg,
    "--data", args.data,
    "--out", result_path]

    subprocess.run(cmd, check=True)
    
    with open(result_path, 'rb') as handle:
        preds_single, meta_image_files_single = pickle.load(handle)
    
    print("Select Predictions:")
    for path, entrie in tqdm(zip(meta_image_files_single, preds_single), total=len(preds_single)): 
        split_path=path[0].split('/')
        Sequence=split_path[-4].split('_')[0]
        Sequence=Sequence.split('e')[-1]
        entrie=select_preds(entrie,ind_dict[Sequence])
        frame=split_path[-1].split('.')[0]
        frame_dict=seq_dict.get(Sequence,{})
        frame_dict.update({int(frame):entrie})
        seq_dict[Sequence]=frame_dict
        image,VisCam=VizualizeAll(path[0],entrie)
        cv2.imwrite(os.path.join(image_path,f"{Sequence}-{frame}.jpg"),image)
    
        
    
    for seq, framedict in seq_dict.items(): 
        print("Match and export Sequence "+str(seq))    
        matched_dict=match_preds(framedict,ind_dict[seq]) 
        export_results(matched_dict,seq,ModelName,OutDir,VisCam) 
        seq_dict[seq]= matched_dict
        
    apply_kalman(OutDir,evalSeqs, ModelName)
    runEvaluation(os.path.abspath(OutDir),DatasetPath, [ModelName], [], evalSeqs)
        
    
