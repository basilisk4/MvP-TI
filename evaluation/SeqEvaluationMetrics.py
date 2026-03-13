"""
Compute metrics on outputs of sequence evaluation (3D + 2D).
Sequential implementation with full CSV summary generation.
"""

import os
import sys
import math
import pickle
import itertools
import statistics

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------

PARENT_DIR = os.path.dirname(os.path.realpath(__file__))
UTILS_DIR = os.path.join(PARENT_DIR, "../Utils")
DATASET_DIR = os.path.join(UTILS_DIR, "Dataset-3DPOP")

sys.path.append(UTILS_DIR)
sys.path.append(DATASET_DIR)

import HungarianAlgorithm
from POP3D_Reader import Trial

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

PIGEON_KEYPOINT_NAMES = [
    "hd_beak", "hd_nose", "hd_leftEye", "hd_rightEye",
    "bp_leftShoulder", "bp_rightShoulder",
    "bp_topKeel", "bp_bottomKeel", "bp_tail"
]

# ---------------------------------------------------------------------
# Simplified basic metrics
# ---------------------------------------------------------------------

def GetEucDist(p1, p2):
    if len(p1) == 3 and len(p2) == 3:
        return math.sqrt(sum((a-b)**2 for a,b in zip(p1,p2)))
    elif len(p1) == 2 and len(p2) == 2:
        return math.sqrt(sum((a-b)**2 for a,b in zip(p1,p2)))
    else:
        raise Exception("Point input size error")

def GetPCK(dist, max_dist):
    return (1 if dist/max_dist < 0.1 else 0, 1 if dist/max_dist < 0.05 else 0)

def GetRMSE(vals):
    vals = [v for v in vals if v == v]
    return np.sqrt(np.mean(np.array(vals)**2))

def GetMedian(vals):
    vals = [v for v in vals if v == v]
    return statistics.median(vals)

def GetPCKSum(vals):
    vals = [v for v in vals if v == v]
    return (sum(vals)/len(vals))*100

# ---------------------------------------------------------------------
# ID matching
# ---------------------------------------------------------------------

def MatchID(sequence, cam, predictions):
    counter = 0
    while True:
        counter += 1
        if counter not in predictions:
            continue

        gt = {bird: list(cam.Read3DKeypointData(cam.Keypoint3D, counter, bird, Keypoints=["bp_bottomKeel"]).values())[0]
              for bird in sequence.Subjects}

        if np.isnan(list(gt.values())).any():
            continue

        try:
            pred = {k.split("_")[0]: v.tolist() for k,v in predictions[counter].items() if "bp_bottomKeel" in k}
        except Exception:
            continue

        dist = cdist(np.array(list(gt.values())), np.array(list(pred.values())))
        matches = HungarianAlgorithm.hungarian_algorithm(dist)
        gt_ids = list(gt.keys())
        pred_ids = list(pred.keys())
        return {gt_ids[m[0]]: pred_ids[m[1]] for m in matches}

# ---------------------------------------------------------------------
# Evaluation (3D + 2D)
# ---------------------------------------------------------------------

def DoEval(dataset_path, seq, preds_3d, preds_2d=None):
    seq_obj = Trial.Trial(dataset_path, seq)
    seq_obj.load3DPopTrainingSet(Filter=True, Type="Test")
    cam0 = seq_obj.camObjects[0]

    matched = MatchID(seq_obj, cam0, preds_3d)

    E3D, P10_3D, P05_3D = [], [], []
    E2D, P10_2D, P05_2D = [], [], []

    for f in tqdm(preds_3d.keys(), desc="Seq "+str(seq)):
        fe3d, f103d, f053d = {}, {}, {}
        fe2d, f102d, f052d = {}, {}, {}

        for gt_id, pred_id in matched.items():
            gt3d = cam0.Read3DKeypointData(cam0.Keypoint3D, f, gt_id, Keypoints=PIGEON_KEYPOINT_NAMES, StripName=True)
            pred3d = {"_".join(k.split("_")[1:3]):v for k,v in preds_3d[f].items() if k.startswith(pred_id)}

            max_dist_3d = max(GetEucDist(a,b) for a,b in itertools.product(gt3d.values(), repeat=2))

            be3d, b103d, b053d = {}, {}, {}
            for kp in PIGEON_KEYPOINT_NAMES:
                if kp not in pred3d or np.isnan(pred3d[kp]).any() or np.isnan(gt3d[kp]).any():
                    d = p10 = p05 = np.nan
                else:
                    d = GetEucDist(gt3d[kp], pred3d[kp])
                    p10, p05 = GetPCK(d, max_dist_3d)
                be3d[kp] = d; b103d[kp] = p10; b053d[kp] = p05
            fe3d[gt_id] = be3d; f103d[gt_id] = b103d; f053d[gt_id] = b053d

            if preds_2d:
                for cam in seq_obj.camObjects:
                    cname = cam.CamName
                    if cname not in preds_2d[f]:
                        continue
                    gt2d = cam.Read2DKeypointData(cam.Keypoint2D, f, gt_id, Keypoints=PIGEON_KEYPOINT_NAMES, StripName=True)
                    pred2d = {"_".join(k.split("_")[1:3]):v for k,v in preds_2d[f][cname].items() if k.startswith(pred_id)}
                    bbox = cam.GetBBoxData(cam.BBox, f, gt_id)
                    max_dist_2d = max(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1])
                    be2d, b102d, b052d = {}, {}, {}
                    for kp in PIGEON_KEYPOINT_NAMES:
                        if kp not in pred2d or np.isnan(pred2d[kp]).any() or np.isnan(gt2d[kp]).any():
                            d = p10 = p05 = np.nan
                        else:
                            d = GetEucDist(gt2d[kp], pred2d[kp])
                            p10, p05 = GetPCK(d, max_dist_2d)
                        be2d[kp] = d; b102d[kp] = p10; b052d[kp] = p05
                    key = f"{cname}_{gt_id}"
                    fe2d[key] = be2d; f102d[key] = b102d; f052d[key] = b052d

        E3D.append(fe3d); P10_3D.append(f103d); P05_3D.append(f053d)
        E2D.append(fe2d); P10_2D.append(f102d); P05_2D.append(f052d)

    return E3D, P10_3D, P05_3D, E2D, P10_2D, P05_2D

# ---------------------------------------------------------------------
# CSV summaries
# ---------------------------------------------------------------------

def RMSESummaryDict(RMSEDictList):
    all_vals, per_kp = [], {k: [] for k in PIGEON_KEYPOINT_NAMES}
    for frame in RMSEDictList:
        for bird in frame.values():
            for k,v in bird.items():
                per_kp[k].append(v)
                all_vals.append(v)
    return per_kp, all_vals

def PCKSummaryDict(PCKDictList):
    all_vals, per_kp = [], {k: [] for k in PIGEON_KEYPOINT_NAMES}
    for frame in PCKDictList:
        for bird in frame.values():
            for k,v in bird.items():
                per_kp[k].append(v)
                all_vals.append(v)
    return per_kp, all_vals

def GetSummaryCSV(EvalDir, Models, Seqs, Type="3D"):
    rows_e, rows_m, rows_10, rows_05 = {}, {}, {}, {}
    idx = 0
    for m in Models:
        all_e, all_10, all_05 = [], [], []
        for s in Seqs:
            all_e += pickle.load(open(os.path.join(EvalDir, f"{m}_Seq{s}_{Type}_EucError.p"), "rb"))
            all_10 += pickle.load(open(os.path.join(EvalDir, f"{m}_Seq{s}_{Type}_PCK10.p"), "rb"))
            all_05 += pickle.load(open(os.path.join(EvalDir, f"{m}_Seq{s}_{Type}_PCK05.p"), "rb"))
        kp_e, vals = RMSESummaryDict(all_e)
        kp10, v10 = PCKSummaryDict(all_10)
        kp05, v05 = PCKSummaryDict(all_05)
        rows_e[idx] = {"Model": m, "Overall": GetRMSE(vals), **{k:GetRMSE(v) for k,v in kp_e.items()}}
        rows_m[idx] = {"Model": m, "Overall": GetMedian(vals), **{k:GetMedian(v) for k,v in kp_e.items()}}
        rows_10[idx] = {"Model": m, "Overall": GetPCKSum(v10), **{k:GetPCKSum(v) for k,v in kp10.items()}}
        rows_05[idx] = {"Model": m, "Overall": GetPCKSum(v05), **{k:GetPCKSum(v) for k,v in kp05.items()}}
        idx += 1
    out = os.path.join(EvalDir, "Seq_EvaluationSummary")
    pd.DataFrame.from_dict(rows_e, orient="index").to_csv(os.path.join(out, f"EucErrorSummary{Type}.csv"))
    pd.DataFrame.from_dict(rows_m, orient="index").to_csv(os.path.join(out, f"MedianSummary{Type}.csv"))
    pd.DataFrame.from_dict(rows_10, orient="index").to_csv(os.path.join(out, f"PCK10Summary{Type}.csv"))
    pd.DataFrame.from_dict(rows_05, orient="index").to_csv(os.path.join(out, f"PCK05Summary{Type}.csv"))
    
def GetIndNumSummaryCSV(EvalDir, Models, AllSequences, Type="3D"):
    EucDFDict, PCK10DFDict, PCK05DFDict, MedianDFDict = {}, {}, {}, {}
    counter = 0

    for model in Models:
        for seq in AllSequences:
            euc = pickle.load(open(os.path.join(EvalDir, f"{model}_Seq{seq}_{Type}_EucError.p"), "rb"))
            pck10 = pickle.load(open(os.path.join(EvalDir, f"{model}_Seq{seq}_{Type}_PCK10.p"), "rb"))
            pck05 = pickle.load(open(os.path.join(EvalDir, f"{model}_Seq{seq}_{Type}_PCK05.p"), "rb"))

            kp_e, all_e = RMSESummaryDict(euc)
            kp_10, all_10 = PCKSummaryDict(pck10)
            kp_05, all_05 = PCKSummaryDict(pck05)

            EucDFDict[counter] = {"Model": model, "Seq": seq, "Overall": GetRMSE(all_e), **{k:GetRMSE(v) for k,v in kp_e.items()}}
            MedianDFDict[counter] = {"Model": model, "Seq": seq, "Overall": GetMedian(all_e), **{k:GetMedian(v) for k,v in kp_e.items()}}
            PCK10DFDict[counter] = {"Model": model, "Seq": seq, "Overall": GetPCKSum(all_10), **{k:GetPCKSum(v) for k,v in kp_10.items()}}
            PCK05DFDict[counter] = {"Model": model, "Seq": seq, "Overall": GetPCKSum(all_05), **{k:GetPCKSum(v) for k,v in kp_05.items()}}
            counter += 1

    out_dir = os.path.join(EvalDir, "Seq_EvaluationSummary")
    pd.DataFrame.from_dict(EucDFDict, orient="index").to_csv(os.path.join(out_dir, f"Ind_EucErrorSummary{Type}.csv"))
    pd.DataFrame.from_dict(PCK10DFDict, orient="index").to_csv(os.path.join(out_dir, f"Ind_PCK10Summary{Type}.csv"))
    pd.DataFrame.from_dict(PCK05DFDict, orient="index").to_csv(os.path.join(out_dir, f"Ind_PCK05Summary{Type}.csv"))
    pd.DataFrame.from_dict(MedianDFDict, orient="index").to_csv(os.path.join(out_dir, f"Ind_MedianSummary{Type}.csv"))
    

# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def runEvaluation(EvalDir, DatasetPath, Models3D, Models2D, Seqs):
    print("Evaluating...")
    os.makedirs(os.path.join(EvalDir, "Seq_EvaluationSummary"), exist_ok=True)
    files = [f for f in os.listdir(EvalDir) if "Kalman3D" in f]
    for f in files:
        _, _, model, seq = f.split("_")
        seq = seq.split(".")[0].replace("Seq","")
        preds3d = pickle.load(open(os.path.join(EvalDir, f), "rb"))
        if model not in Models2D:
            e3d, p103d, p053d, _, _, _ = DoEval(DatasetPath, seq, preds3d)
            pickle.dump(e3d, open(os.path.join(EvalDir, f"{model}_Seq{seq}_3D_EucError.p"), "wb"))
            pickle.dump(p103d, open(os.path.join(EvalDir, f"{model}_Seq{seq}_3D_PCK10.p"), "wb"))
            pickle.dump(p053d, open(os.path.join(EvalDir, f"{model}_Seq{seq}_3D_PCK05.p"), "wb"))
            continue
        preds2d = pickle.load(open(os.path.join(EvalDir, f"SeqEval_Points2D_{model}_Seq{seq}.p"), "rb"))
        e3d, p103d, p053d, e2d, p102d, p052d = DoEval(DatasetPath, seq, preds3d, preds2d)
        pickle.dump(e3d, open(os.path.join(EvalDir, f"{model}_Seq{seq}_3D_EucError.p"), "wb"))
        pickle.dump(p103d, open(os.path.join(EvalDir, f"{model}_Seq{seq}_3D_PCK10.p"), "wb"))
        pickle.dump(p053d, open(os.path.join(EvalDir, f"{model}_Seq{seq}_3D_PCK05.p"), "wb"))
        pickle.dump(e2d, open(os.path.join(EvalDir, f"{model}_Seq{seq}_2D_EucError.p"), "wb"))
        pickle.dump(p102d, open(os.path.join(EvalDir, f"{model}_Seq{seq}_2D_PCK10.p"), "wb"))
        pickle.dump(p052d, open(os.path.join(EvalDir, f"{model}_Seq{seq}_2D_PCK05.p"), "wb"))

    GetSummaryCSV(EvalDir, Models3D, Seqs, Type="3D")
    GetIndNumSummaryCSV(EvalDir,Models3D,Seqs, Type = "3D")
    GetSummaryCSV(EvalDir, Models2D, Seqs, Type="2D")
    GetIndNumSummaryCSV(EvalDir,Models2D,Seqs, Type = "2D")

# ---------------------------------------------------------------------
