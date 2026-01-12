"""Compute metrics on outputs of sequence evaluation"""

import numpy as np
import os
import sys

ParentDir=os.path.dirname(os.path.realpath(__file__))
path=os.path.join(ParentDir,"../Utils/")
sys.path.append(path)
path=os.path.join(path,"Dataset-3DPOP")
import HungarianAlgorithm
sys.path.append(path)
from POP3D_Reader import Trial

import HungarianAlgorithm

import pickle
import math
from tqdm import tqdm
import itertools
from glob import glob

from scipy.spatial.distance import cdist
import statistics
import pandas as pd



PIGEON_KEYPOINT_NAMES = ["hd_beak","hd_nose","hd_leftEye","hd_rightEye","bp_leftShoulder","bp_rightShoulder","bp_topKeel","bp_bottomKeel","bp_tail"]


def GetEucDist(Point1,Point2):
    """Get euclidian error, both 2D and 3D"""

    if len(Point1) ==3 & len(Point2) ==3:
        EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2) + ((Point1[2] - Point2[2]) ** 2) )
    elif len(Point1) ==2 & len(Point2) ==2:
        EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2))
    else:
        import ipdb;ipdb.set_trace()
        Exception("point input size error")

    return EucDist

def GetPCK(PointDist,MaxDist):
    """Given euclidiean distance between 2 points and max distance between 2 points of given pigeon, calc whether keypoint is correct, as 1 and 0"""
    PCK10 = 0
    PCK05 = 0

    PercentageofMax = PointDist/MaxDist

    if PercentageofMax < 0.1:
        PCK10 = 1

    if PercentageofMax < 0.05:
        PCK05 = 1

    return PCK10, PCK05

def GetMedian(ErrorList):
    """Given List of errors, calculate RMSE"""
    ErrorList = [x for x in ErrorList if x == x] #trick to remove NANs, because nan != nan

    Out = statistics.median(ErrorList)

    return Out


def GetPCKSum(PCKList):
    PCKList = [x for x in PCKList if x == x]

    return (sum(PCKList)/len(PCKList))*100


def GetRMSE(ErrorList):
    """Given List of errors, calculate RMSE"""
    ErrorList = [x for x in ErrorList if x == x] #trick to remove NANs, because nan != nan
    # import ipdb;ipdb.set_trace()
    Out = np.sqrt(np.mean(np.array(ErrorList)**2))
    # Out = np.mean(np.array(ErrorList))
    return Out

def MatchID(SequenceObj,CamObj, Predictions, counter = None):
    """Given tracking output IDS (0-9) and bird IDs from 3D POP, find a frame and match them up"""
    # import ipdb;ipdb.set_trace()
    if counter == None:
        counter = 0

    MatchedDict = {}
    while True:
        # print(counter)
        if counter <5:
            counter += 1
            continue


        if counter not in Predictions:
            counter +=1
            continue

        FramePred = Predictions[counter]
        GTDict = {}
        for bird in SequenceObj.Subjects:
            GTDict[bird] = list(CamObj.Read3DKeypointData(CamObj.Keypoint3D, counter, bird, Keypoints = ["bp_bottomKeel"]).values())[0]
        if np.isnan(list(GTDict.values())).any(): ##Some of the points is nan
            counter += 1
            continue

        # import ipdb;ipdb.set_trace()
        try:
            PredDict = {k.split("_")[0]:v.tolist() for k,v in FramePred.items() if "bp_bottomKeel" in k}
        except:
            counter +=1
            continue

        PredNP = np.array(list(PredDict.values()))
        GTNP = np.array(list(GTDict.values()))

        DistanceMatrix = cdist(GTNP, PredNP)

        # HungarianAlgorithm
        Matches = HungarianAlgorithm.hungarian_algorithm(DistanceMatrix)
        BirdIDs = list(GTDict.keys())
        PredIDs = list(PredDict.keys())

        MatchedDict = {}
        for match in Matches:
            MatchedDict[BirdIDs[match[0]]] = PredIDs[match[1]]

        break

    return MatchedDict

def RMSESummaryDict(RMSEDictList,Keypoints, filter = False):
    """Given dictionary of euc errors, process and get dict of errors"""

    ##Prepare dictionary for per keypoint error:
    PerKeypointDict = {}
    for key in Keypoints:
        PerKeypointDict[key] = []

    AllPointsList = []
    # import ipdb;ipdb.set_trace()
    IndividualFilterCounter = 0
    TotalIndividualsCounter = 0

    # import ipdb;ipdb.set_trace()
    for i in range(len(RMSEDictList)):
        FrameDict = RMSEDictList[i]

        for PointsDict in FrameDict.values():
            TotalIndividualsCounter += 1
            MeanVal = np.array(list(PointsDict.values())).mean()
            if filter:
                if MeanVal > filter:
                    IndividualFilterCounter += 1
                    continue

            for k,v in PointsDict.items():
                PerKeypointDict[k].append(v)
                AllPointsList.append(v)

    #print(IndividualFilterCounter)
    #print("Total Individuals: %s"%TotalIndividualsCounter)

    return PerKeypointDict, AllPointsList


def RMSESummary(RMSEDictList,Keypoints):
    """Given dictionary of euc errors, process and print RMSE MPJEs"""

    ##Prepare dictionary for per keypoint error:
    PerKeypointDict = {}
    for key in Keypoints:
        PerKeypointDict[key] = []

    AllPointsList = []
    for i in range(len(RMSEDictList)):
        FrameDict = RMSEDictList[i]

        for PointsDict in FrameDict.values():
            for k,v in PointsDict.items():
                PerKeypointDict[k].append(v)
                AllPointsList.append(v)

    ##Calc RMSE
    for Key,Val in PerKeypointDict.items():
        RMSE = GetRMSE(Val)
        print(Key + ":")
        print(RMSE)

    print("Overall RMSE:")
    print(GetRMSE(AllPointsList))


def PCKSummary(PCKDictList, Keypoints):
    """Given dictionary of PCKs, process and print PCK"""
    PerKeypointDict = {}
    for key in Keypoints:
        PerKeypointDict[key] = []

    AllPointsList = []
    for i in range(len(PCKDictList)):
        FrameDict = PCKDictList[i]

        for PointsDict in FrameDict.values():
            for k,v in PointsDict.items():
                PerKeypointDict[k].append(v)
                AllPointsList.append(v)
    ##Calc PCK
    for Key,Val in PerKeypointDict.items():
        PCK = GetPCKSum(Val)
        print(Key + ":")
        print(PCK)

    print("Overall PCK:")
    print( GetPCKSum(AllPointsList))


def DoEval3D(DatasetPath, SeqNum,Predictions3D):
    """do evaluation, only for 3D"""
    SequenceObj = Trial.Trial(DatasetPath,SeqNum)
    SequenceObj.load3DPopTrainingSet(Filter = True, Type = "Test")
    CamObj = SequenceObj.camObjects[0] #Just use one of the cam objects
    FrameNums = Predictions3D.keys()

    #Match corresponding IDs
    # import ipdb;ipdb.set_trace()
    MatchedDict = MatchID(SequenceObj,CamObj, Predictions3D)
    EucErrorList3D = []
    PCK05List3D = []
    PCK10List3D = []


    for i in tqdm(FrameNums):
        # MatchedDict = MatchID(SequenceObj,CamObj, Predictions3D,i)

        # GTDict = CamObj.Load3DKeypoint(CamObj.Keypoint3D, i)
        FramePred3D = Predictions3D[i]
        EucErrorDict3D = {}
        PCK05Dict3D = {}
        PCK10Dict3D = {}


        for GTID, PredID in MatchedDict.items(): #for each bird
            Bird3DGT = CamObj.Read3DKeypointData(CamObj.Keypoint3D, i, GTID,Keypoints = PIGEON_KEYPOINT_NAMES,StripName=True)
            #print(Bird3DGT)
            Bird3DPred = {"_".join(k.split("_")[1:3]):v for k,v in FramePred3D.items() if k.startswith(PredID)}

            #### DO 3D EVALUATION
            ##Find max distance between any keypoints for PCK
            DistList = []
            for pair in itertools.product(list(Bird3DGT.values()),repeat=2):
                DistList.append(GetEucDist(pair[0],pair[1]))

            MaxDist = max(DistList)
            BirdEucErrorDict3D = {}
            BirdPCK10Dict3D = {}
            BirdPCK05Dict3D = {}

            for kp in PIGEON_KEYPOINT_NAMES:

                if np.isnan(np.array(list(Bird3DGT.values()))).any(): ##if there is any nan
                    # print("wow")
                    PointDist = np.nan
                    PCK10 = np.nan
                    PCK05 = np.nan
                elif kp not in Bird3DPred:
                    PointDist = np.nan
                    PCK10 = np.nan
                    PCK05 = np.nan
                else:
                    GTval = Bird3DGT[kp]
                    PredVal = Bird3DPred[kp]
                    # import ipdb;ipdb.set_trace()
                    if np.isnan(PredVal).any() or np.isnan(GTval).any():
                        PointDist = np.nan
                        PCK10 = np.nan
                        PCK05 = np.nan
                    else:
                        PointDist = GetEucDist(GTval,PredVal)
                        PCK10,PCK05 = GetPCK(PointDist,MaxDist)


                BirdEucErrorDict3D[kp] = PointDist
                BirdPCK10Dict3D[kp] = PCK10
                BirdPCK05Dict3D[kp] = PCK05

            EucErrorDict3D[GTID] = BirdEucErrorDict3D
            PCK10Dict3D[GTID] = BirdPCK10Dict3D
            PCK05Dict3D[GTID] = BirdPCK05Dict3D

        # import ipdb;ipdb.set_trace()

        EucErrorList3D.append(EucErrorDict3D)
        PCK10List3D.append(PCK10Dict3D)
        PCK05List3D.append(PCK05Dict3D)


    return EucErrorList3D,PCK10List3D,PCK05List3D


def DoEval(DatasetPath, SeqNum,Predictions3D,Predictions2D):
    """do evaluation for 3D"""
    SequenceObj = Trial.Trial(DatasetPath,SeqNum)
    SequenceObj.load3DPopTrainingSet(Filter = True, Type = "Test")
    CamObj = SequenceObj.camObjects[0] #Just use one of the cam objects
    FrameNums = Predictions3D.keys()

    #Match corresponding IDs
    MatchedDict = MatchID(SequenceObj,CamObj, Predictions3D)
    EucErrorList3D = []
    PCK05List3D = []
    PCK10List3D = []
    EucErrorList2D = []
    PCK05List2D = []
    PCK10List2D = []



    for i in tqdm(FrameNums):
        # GTDict = CamObj.Load3DKeypoint(CamObj.Keypoint3D, i)
        FramePred3D = Predictions3D[i]
        FramePred2D = Predictions2D[i]
        EucErrorDict3D = {}
        PCK05Dict3D = {}
        PCK10Dict3D = {}
        EucErrorDict2D = {}
        PCK05Dict2D = {}
        PCK10Dict2D = {}

        for GTID, PredID in MatchedDict.items(): #for each bird
            Bird3DGT = CamObj.Read3DKeypointData(CamObj.Keypoint3D, i, GTID,Keypoints = PIGEON_KEYPOINT_NAMES,StripName=True)
            Bird3DPred = {"_".join(k.split("_")[1:3]):v for k,v in FramePred3D.items() if k.startswith(PredID)}

            #### DO 3D EVALUATION
            ##Find max distance between any keypoints for PCK
            DistList = []
            for pair in itertools.product(list(Bird3DGT.values()),repeat=2):
                DistList.append(GetEucDist(pair[0],pair[1]))

            MaxDist = max(DistList)
            BirdEucErrorDict3D = {}
            BirdPCK10Dict3D = {}
            BirdPCK05Dict3D = {}

            for kp in PIGEON_KEYPOINT_NAMES:

                if kp not in Bird3DPred:
                    PointDist = np.nan
                    PCK10 = np.nan
                    PCK05 = np.nan
                else:
                    GTval = Bird3DGT[kp]
                    PredVal = Bird3DPred[kp]
                    # import ipdb;ipdb.set_trace()
                    if np.isnan(PredVal).any() or np.isnan(GTval).any():
                        PointDist = np.nan
                        PCK10 = np.nan
                        PCK05 = np.nan
                    else:
                        PointDist = GetEucDist(GTval,PredVal)
                        PCK10,PCK05 = GetPCK(PointDist,MaxDist)

                # if PointDist > 1000:
                #     print(SeqNum)
                #     import ipdb;ipdb.set_trace()


                BirdEucErrorDict3D[kp] = PointDist
                BirdPCK10Dict3D[kp] = PCK10
                BirdPCK05Dict3D[kp] = PCK05


            ### DO 2D EVALUATION
            for camObj in SequenceObj.camObjects:
                CamName = camObj.CamName
                if CamName not in FramePred2D:
                    continue

                Bird2DGT = camObj.Read2DKeypointData(camObj.Keypoint2D, i, GTID,Keypoints = PIGEON_KEYPOINT_NAMES,StripName=True)


                Bird2DPred = {"_".join(k.split("_")[1:3]):v for k,v in FramePred2D[CamName].items() if k.startswith(PredID)}
                Bird2DBBox = camObj.GetBBoxData(camObj.BBox ,i,GTID )

                ##Max dimension of bbox
                MaxDist = max([Bird2DBBox[1][0]-Bird2DBBox[0][0], Bird2DBBox[1][1]-Bird2DBBox[0][1]])
                BirdEucErrorDict2D = {}
                BirdPCK10Dict2D = {}
                BirdPCK05Dict2D = {}

                for kp in PIGEON_KEYPOINT_NAMES:
                    if kp not in Bird2DPred:
                        PointDist = np.nan
                        PCK10 = np.nan
                        PCK05 = np.nan
                    else:
                        GTval = Bird2DGT[kp]
                        PredVal = Bird2DPred[kp]
                        # import ipdb;ipdb.set_trace()
                        if np.isnan(PredVal).any() or np.isnan(GTval).any():
                            PointDist = np.nan
                            PCK10 = np.nan
                            PCK05 = np.nan
                        else:
                            PointDist = GetEucDist(GTval,PredVal)
                            PCK10,PCK05 = GetPCK(PointDist,MaxDist)



                    BirdEucErrorDict2D[kp] = PointDist
                    BirdPCK10Dict2D[kp] = PCK10
                    BirdPCK05Dict2D[kp] = PCK05

                EucErrorDict2D["%s_%s"%(CamName,GTID)] = BirdEucErrorDict2D
                PCK10Dict2D["%s_%s"%(CamName,GTID)] = BirdPCK10Dict2D
                PCK05Dict2D["%s_%s"%(CamName,GTID)] = BirdPCK05Dict2D


            EucErrorDict3D[GTID] = BirdEucErrorDict3D
            PCK10Dict3D[GTID] = BirdPCK10Dict3D
            PCK05Dict3D[GTID] = BirdPCK05Dict3D


        # import ipdb;ipdb.set_trace()

        EucErrorList3D.append(EucErrorDict3D)
        PCK10List3D.append(PCK10Dict3D)
        PCK05List3D.append(PCK05Dict3D)

        EucErrorList2D.append(EucErrorDict2D)
        PCK10List2D.append(PCK10Dict2D)
        PCK05List2D.append(PCK05Dict2D)


    return EucErrorList3D,PCK10List3D,PCK05List3D, EucErrorList2D, PCK10List2D,PCK05List2D

def PCKSummaryDict(PCKDictList, Keypoints):
    """Given dictionary of PCKs, process and return PCK dicts"""
    PerKeypointDict = {}
    for key in Keypoints:
        PerKeypointDict[key] = []

    AllPointsList = []
    for i in range(len(PCKDictList)):
        FrameDict = PCKDictList[i]

        for PointsDict in FrameDict.values():
            for k,v in PointsDict.items():
                PerKeypointDict[k].append(v)
                AllPointsList.append(v)

    return PerKeypointDict, AllPointsList

def GetSummaryCSV(EvalDir,Models,AllSequences, Type = "3D"):
    EucDFDict = {}
    PCK10DFDict = {}
    PCK05DFDict = {}
    MedianDFDict = {}
    counter = 0

    # import ipdb;ipdb.set_trace()
    for ModelName in Models:
        AllEucErrorList = []
        AllPCK10List = []
        AllPCK05List = []

        for SeqNum in AllSequences:
            if Type == "3D":
                EucErrorList = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_EucError.p"%(ModelName,SeqNum)),"rb"))
                PCK10List = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK10.p"%(ModelName,SeqNum)),"rb"))
                PCK05List = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK05.p"%(ModelName,SeqNum)),"rb"))
            elif Type == "2D":
                EucErrorList = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_2D_EucError.p"%(ModelName,SeqNum)),"rb"))
                PCK10List = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK10.p"%(ModelName,SeqNum)),"rb"))
                PCK05List = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK05.p"%(ModelName,SeqNum)),"rb"))

            # print(EucErrorList)
            # import ipdb;ipdb.set_trace()

            AllEucErrorList.extend(EucErrorList)
            AllPCK10List.extend(PCK10List)
            AllPCK05List.extend(PCK05List)

        ###Save to CSV
        ModelEucDict = {"Model":ModelName}
        PerKeypointDictEuc, AllPointsListEuc = RMSESummaryDict(AllEucErrorList,PIGEON_KEYPOINT_NAMES,filter=False)
        ModelEucDict["Overall"] = GetRMSE(AllPointsListEuc)
        for Key,Val in PerKeypointDictEuc.items():
            RMSE = GetRMSE(Val)
            ModelEucDict.update({Key:RMSE})
        EucDFDict[counter] = ModelEucDict

        ##Median
        ModelMedDict = {"Model":ModelName}
        PerKeypointDictEuc, AllPointsListEuc = RMSESummaryDict(AllEucErrorList,PIGEON_KEYPOINT_NAMES,filter=False)
        ModelMedDict.update({"Overall":GetMedian(AllPointsListEuc)})
        for Key,Val in PerKeypointDictEuc.items():
            RMSE = GetMedian(Val)
            ModelMedDict.update({Key:RMSE})
        MedianDFDict[counter]= ModelMedDict

        #PCK10:
        ModelPCK10Dict = {"Model":ModelName}
        PerKeypointDictPCK10, AllPointsListPCK10 = PCKSummaryDict(AllPCK10List,PIGEON_KEYPOINT_NAMES)
        ModelPCK10Dict.update({"Overall":GetPCKSum(AllPointsListPCK10)})
        for Key,Val in PerKeypointDictPCK10.items():
            PCK = GetPCKSum(Val)
            ModelPCK10Dict.update({Key:PCK})
        PCK10DFDict[counter] = ModelPCK10Dict

        #PCK05
        ModelPCK05Dict = {"Model":ModelName}
        PerKeypointDictPCK05, AllPointsListPCK05 = PCKSummaryDict(AllPCK05List,PIGEON_KEYPOINT_NAMES)
        ModelPCK05Dict.update({"Overall":GetPCKSum(AllPointsListPCK05)})
        for Key,Val in PerKeypointDictPCK05.items():
            PCK = GetPCKSum(Val)
            ModelPCK05Dict.update({Key:PCK})
        PCK05DFDict[counter] = ModelPCK05Dict

        counter += 1

    # import ipdb;ipdb.set_trace()
    EucDF = pd.DataFrame.from_dict(EucDFDict,orient= "index")
    EucDF.to_csv(os.path.join(EvalDir,"Seq_EvaluationSummary/EucErrorSummary%s.csv"%Type))

    PCK10DF = pd.DataFrame.from_dict(PCK10DFDict,orient= "index")
    PCK10DF.to_csv(os.path.join(EvalDir,"Seq_EvaluationSummary/PCK10Summary%s.csv"%Type))

    PCK05DF = pd.DataFrame.from_dict(PCK05DFDict,orient= "index")
    PCK05DF.to_csv(os.path.join(EvalDir,"Seq_EvaluationSummary/PCK05Summary%s.csv"%Type))

    MedianDF = pd.DataFrame.from_dict(MedianDFDict,orient= "index")
    MedianDF.to_csv(os.path.join(EvalDir,"Seq_EvaluationSummary/MedianSummary%s.csv"%Type))

def GetIndNumSummaryCSV(EvalDir,Models,AllSequences, Type = "3D"):
    EucDFDict = {}
    PCK10DFDict = {}
    PCK05DFDict = {}
    MedianDFDict = {}
    counter = 0

    # import ipdb;ipdb.set_trace()
    for ModelName in Models:
        AllEucErrorList = []
        AllPCK10List = []
        AllPCK05List = []

        for SeqNum in AllSequences:
            if Type == "3D":
                EucErrorList = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_EucError.p"%(ModelName,SeqNum)),"rb"))
                PCK10List = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK10.p"%(ModelName,SeqNum)),"rb"))
                PCK05List = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK05.p"%(ModelName,SeqNum)),"rb"))
            elif Type == "2D":
                EucErrorList = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_2D_EucError.p"%(ModelName,SeqNum)),"rb"))
                PCK10List = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK10.p"%(ModelName,SeqNum)),"rb"))
                PCK05List = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK05.p"%(ModelName,SeqNum)),"rb"))

            # print(EucErrorList)
            # import ipdb;ipdb.set_trace()

            AllEucErrorList = EucErrorList
            AllPCK10List = PCK10List
            AllPCK05List = PCK05List

            ###Save to CSV
            ModelEucDict = {"Model":ModelName,"Seq":SeqNum}
            PerKeypointDictEuc, AllPointsListEuc = RMSESummaryDict(AllEucErrorList,PIGEON_KEYPOINT_NAMES,filter=False)
            ModelEucDict["Overall"] = GetRMSE(AllPointsListEuc)
            for Key,Val in PerKeypointDictEuc.items():
                RMSE = GetRMSE(Val)
                ModelEucDict.update({Key:RMSE})
            EucDFDict[counter] = ModelEucDict

            ##Median
            ModelMedDict = {"Model":ModelName, "Seq":SeqNum}
            PerKeypointDictEuc, AllPointsListEuc = RMSESummaryDict(AllEucErrorList,PIGEON_KEYPOINT_NAMES,filter=False)
            ModelMedDict.update({"Overall":GetMedian(AllPointsListEuc)})
            for Key,Val in PerKeypointDictEuc.items():
                RMSE = GetMedian(Val)
                ModelMedDict.update({Key:RMSE})
            MedianDFDict[counter]= ModelMedDict

            #PCK10:
            ModelPCK10Dict = {"Model":ModelName, "Seq":SeqNum}
            PerKeypointDictPCK10, AllPointsListPCK10 = PCKSummaryDict(AllPCK10List,PIGEON_KEYPOINT_NAMES)
            ModelPCK10Dict.update({"Overall":GetPCKSum(AllPointsListPCK10)})
            for Key,Val in PerKeypointDictPCK10.items():
                PCK = GetPCKSum(Val)
                ModelPCK10Dict.update({Key:PCK})
            PCK10DFDict[counter] = ModelPCK10Dict

            #PCK05
            ModelPCK05Dict = {"Model":ModelName, "Seq":SeqNum}
            PerKeypointDictPCK05, AllPointsListPCK05 = PCKSummaryDict(AllPCK05List,PIGEON_KEYPOINT_NAMES)
            ModelPCK05Dict.update({"Overall":GetPCKSum(AllPointsListPCK05)})
            for Key,Val in PerKeypointDictPCK05.items():
                PCK = GetPCKSum(Val)
                ModelPCK05Dict.update({Key:PCK})
            PCK05DFDict[counter] = ModelPCK05Dict

            counter += 1

    # import ipdb;ipdb.set_trace()
    EucDF = pd.DataFrame.from_dict(EucDFDict,orient= "index")
    EucDF.to_csv(os.path.join(EvalDir,"Seq_EvaluationSummary/Ind_EucErrorSummary%s.csv"%Type))

    PCK10DF = pd.DataFrame.from_dict(PCK10DFDict,orient= "index")
    PCK10DF.to_csv(os.path.join(EvalDir,"Seq_EvaluationSummary/Ind_PCK10Summary%s.csv"%Type))

    PCK05DF = pd.DataFrame.from_dict(PCK05DFDict,orient= "index")
    PCK05DF.to_csv(os.path.join(EvalDir,"Seq_EvaluationSummary/Ind_PCK05Summary%s.csv"%Type))

    MedianDF = pd.DataFrame.from_dict(MedianDFDict,orient= "index")
    MedianDF.to_csv(os.path.join(EvalDir,"Seq_EvaluationSummary/Ind_MedianSummary%s.csv"%Type))


def RunParallel(EvalDir, DatasetPath,file3d):
    _, _, ModelName, SeqNum= file3d.split("_")
    SeqNum = SeqNum.split(".")[0].split("Seq")[1]

    print("Seq: %s, Model: %s"%(SeqNum,ModelName))

    # if os.path.exists(os.path.join(EvalDir,"%s_Seq%s_3D_EucError.p"%(ModelName,SeqNum))):
    #     return False

    if ModelName == "ltohp" or ModelName == "MvP":
        Predictions3D= pickle.load(open(os.path.join(EvalDir,file3d ), "rb"))
        EucErrorList3D,PCK10List3D,PCK05List3D = DoEval3D(DatasetPath, SeqNum,Predictions3D)
        pickle.dump(EucErrorList3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_EucError.p"%(ModelName,SeqNum)),"wb"))
        pickle.dump(PCK10List3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK10.p"%(ModelName,SeqNum)),"wb"))
        pickle.dump(PCK05List3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK05.p"%(ModelName,SeqNum)),"wb"))

        return True

    Predictions3D= pickle.load(open(os.path.join(EvalDir,file3d ), "rb"))
    Predictions2D = pickle.load(open(os.path.join(EvalDir,"SeqEval_Points2D_%s_Seq%s.pkl"%(ModelName,SeqNum) ), "rb"))

    EucErrorList3D,PCK10List3D,PCK05List3D, EucErrorList2D, PCK10List2D,PCK05List2D = DoEval(DatasetPath, SeqNum,Predictions3D,Predictions2D)

    pickle.dump(EucErrorList3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_EucError.p"%(ModelName,SeqNum)),"wb"))
    pickle.dump(PCK10List3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK10.p"%(ModelName,SeqNum)),"wb"))
    pickle.dump(PCK05List3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK05.p"%(ModelName,SeqNum)),"wb"))

    pickle.dump(EucErrorList2D, open(os.path.join(EvalDir,"%s_Seq%s_2D_EucError.p"%(ModelName,SeqNum)),"wb"))
    pickle.dump(PCK10List2D, open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK10.p"%(ModelName,SeqNum)),"wb"))
    pickle.dump(PCK05List2D, open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK05.p"%(ModelName,SeqNum)),"wb"))


    return True


def runEvaluation(EvalDir,DatasetPath, Models3D, Models2D, AllSequences):
    
    os.mkdir(os.path.join(EvalDir,"Seq_EvaluationSummary"))

    PickleFiles = [os.path.basename(file) for file in glob(EvalDir + "/*.p")]
    #### 3D + 2D evaluation:
    Files3D = [file for file in PickleFiles if "RollingFilter3D" in file]
    # Files3D = [file for file in PickleFiles if "Filtered3D" in file]
    # import ipdb;ipdb.set_trace()
    for file3d in Files3D:
        RunParallel(EvalDir,DatasetPath,file3d)


    # # import ipdb;ipdb.set_trace()
    for file3d in Files3D:
        # file3d = Files3D[len(Files3D)-1]
        _, _, ModelName, SeqNum= file3d.split("_")
        SeqNum = SeqNum.split(".")[0].split("Seq")[1]

        print("Seq: %s, Model: %s"%(SeqNum,ModelName))

        # if os.path.exists(os.path.join(EvalDir,"%s_Seq%s_3D_EucError.p"%(ModelName,SeqNum))):
            # continue
        if ModelName == "ltohp" or ModelName == "MvP":
            Predictions3D= pickle.load(open(os.path.join(EvalDir,file3d ), "rb"))
            EucErrorList3D,PCK10List3D,PCK05List3D = DoEval3D(DatasetPath, SeqNum,Predictions3D)
            pickle.dump(EucErrorList3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_EucError.p"%(ModelName,SeqNum)),"wb"))
            pickle.dump(PCK10List3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK10.p"%(ModelName,SeqNum)),"wb"))
            pickle.dump(PCK05List3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK05.p"%(ModelName,SeqNum)),"wb"))

            continue


        Predictions3D= pickle.load(open(os.path.join(EvalDir,file3d ), "rb"))
        Predictions2D = pickle.load(open(os.path.join(EvalDir,"SeqEval_Points2D_%s_Seq%s.p"%(ModelName,SeqNum) ), "rb"))

        EucErrorList3D,PCK10List3D,PCK05List3D, EucErrorList2D, PCK10List2D,PCK05List2D = DoEval(DatasetPath, SeqNum,Predictions3D,Predictions2D)

        pickle.dump(EucErrorList3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_EucError.p"%(ModelName,SeqNum)),"wb"))
        pickle.dump(PCK10List3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK10.p"%(ModelName,SeqNum)),"wb"))
        pickle.dump(PCK05List3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK05.p"%(ModelName,SeqNum)),"wb"))

        pickle.dump(EucErrorList2D, open(os.path.join(EvalDir,"%s_Seq%s_2D_EucError.p"%(ModelName,SeqNum)),"wb"))
        pickle.dump(PCK10List2D, open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK10.p"%(ModelName,SeqNum)),"wb"))
        pickle.dump(PCK05List2D, open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK05.p"%(ModelName,SeqNum)),"wb"))

    
    GetSummaryCSV(EvalDir,Models3D,AllSequences, Type = "3D")
    GetIndNumSummaryCSV(EvalDir,Models3D,AllSequences, Type = "3D")
    
    if not(ModelName == "ltohp" or ModelName == "MvP"):
        GetSummaryCSV(EvalDir,Models2D,AllSequences, Type = "2D")
        GetIndNumSummaryCSV(EvalDir,Models2D,AllSequences, Type = "2D")
    
    

if __name__ == "__main__":
    EvalDir = "/media/alexchan/Extreme SSD/WorkDir/Pigeon3DTrack/SeqEvaluation"
    DatasetPath = "/media/alexchan/My Passport/Dataset_3DPOP"
    ###Get summary Data Frames
    Models3D = ["YOLOVit","KPRCNNSingle","YOLOPose","YOLODLC","KPRCNN","ltohp"]
    Models2D = ["YOLOVit","KPRCNNSingle","YOLOPose","YOLODLC","KPRCNN"]
    AllSequences = [11,1,2,5] #
    
    runEvaluation(EvalDir,DatasetPath, Models3D, Models2D, AllSequences)

    ###Get Per individual Num summary



    # for file3d in Files3D:
    #     _, _, ModelName, SeqNum= file3d.split("_")
    #     SeqNum = SeqNum.split(".")[0].split("Seq")[1]
    #     if ModelName == "ltohp":
    #         continue

    #     print("Seq: %s, Model: %s"%(SeqNum,ModelName))

    #     EucErrorList3D = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_EucError.p"%(ModelName,SeqNum)),"rb"))
    #     PCK10List3D = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK10.p"%(ModelName,SeqNum)),"rb"))
    #     PCK05List3D = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK05.p"%(ModelName,SeqNum)),"rb"))

    #     EucErrorList2D = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_2D_EucError.p"%(ModelName,SeqNum)),"rb"))
    #     PCK10List2D = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK10.p"%(ModelName,SeqNum)),"rb"))
    #     PCK05List2D = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK05.p"%(ModelName,SeqNum)),"rb"))

    #     print("RMSE")
    #     RMSESummary(EucErrorList3D,PIGEON_KEYPOINT_NAMES)
    #     print("PCK10")
    #     PCKSummary(PCK10List3D, PIGEON_KEYPOINT_NAMES)
    #     print("PCK05")
    #     PCKSummary(PCK05List3D, PIGEON_KEYPOINT_NAMES)


