"""Run sequence with kalman filter"""

import argparse
import numpy as np
import pickle
import math
from tqdm import tqdm
from glob import glob
import os

import pandas as pd

from pykalman import KalmanFilter
from natsort import natsort
import copy

###write a function to calculate the euclidian distance between two points

def EucDist(Point1,Point2):
    """Get euclidian error, both 2D and 3D"""
    
    if len(Point1) ==3 & len(Point2) ==3:
        EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2) + ((Point1[2] - Point2[2]) ** 2) )
    elif len(Point1) ==2 & len(Point2) ==2:
        EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2))
    else:
        import ipdb;ipdb.set_trace()
        Exception("point input size error")
    
    return EucDist


def RunRollingAverage(ColVals, WindowSize = 10):
    # import ipdb;ipdb.set_trace()
    Detections = copy.deepcopy(ColVals)
    # Output = copy.deepcopy(ColVals)

    for i in range(ColVals.shape[0]):

        if i < WindowSize:
            continue
        else:
            # Avg = np.nanmean(ColVals[i-WindowSize:i],axis = 0)
            Avg = np.nanmean(Detections[i-WindowSize:i],axis = 0)

            # import ipdb;ipdb.set_trace()

            if abs(EucDist(Avg,ColVals[i])) > 10:
                ColVals[i] = np.array([np.nan,np.nan,np.nan])
            # if any(abs(ColVals[i]-Avg)>10):
            #     ColVals[i] = np.array([np.nan,np.nan,np.nan])

    # # 
    # plt.figure(1)
    # times = range(ColVals.shape[0])
    # plt.plot(times, Detections[:, 0], 'bo',
    #         times, Detections[:, 1], 'ro',
    #         times, Detections[:, 2], 'go',
    #         times, ColVals[:,0], 'b--',
    #         times, ColVals[:,1], 'r--',
    #         times, ColVals[:, 2], 'g--',
    #         markersize=1, linewidth=2)
    # plt.show()
    # import ipdb;ipdb.set_trace()



    return ColVals



def RunKalman(ColVals):
    """    
    Run kalman filter to smooth points?
    Referencing this: https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data
    """

    ##Initial State mean:
    InitialStateList = []
    FirstValIndex = np.where(~np.isnan(ColVals))[0][0]
    for col in range(ColVals.shape[1]):
        InitialStateList.append(ColVals[FirstValIndex,col])


    initial_state_mean = [InitialStateList[0],0,InitialStateList[1],0,InitialStateList[2],0]

    transition_matrix = [[1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0]]


    ##Masked Array:
    # import ipdb;ipdb.set_trace()
    MaskedData = np.ma.array(ColVals, mask = np.isnan(ColVals))

    # Process noise covariance (increase these values)
    process_noise_cov = np.eye(6) * 0.99
    # process_noise_cov[(1,3,5),:] = process_noise_cov[(1,3,5),:] * 2

    # Observation noise covariance (decrease this value)
    observation_noise_cov = np.array([[1],[1],[1]])  


    # kf1 = KalmanFilter(transition_matrices = transition_matrix,
    #                 observation_matrices = observation_matrix,
    #                 initial_state_mean = initial_state_mean,
    #                 initial_state_covariance = np.eye(6),
    #                   transition_covariance=process_noise_cov,
    #                   observation_covariance=observation_noise_cov)


    kf1 = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean,
                    # initial_state_covariance = np.eye(6),
                    # transition_covariance=process_noise_cov,
                    # observation_covariance=observation_noise_cov
                    )


    # kf1 = kf1.em(MaskedData, n_iter=5)


    #initialize:
    # import ipdb;ipdb.set_trace()
    # kf1.transition_covariance[(1,3,5),:] = kf1.transition_covariance[(1,3,5),:]*2

    InitializeNum = 2
    kf1 = kf1.em(MaskedData[:InitializeNum], n_iter=5)

    (filtered_state_means, filtered_state_covariances)  = kf1.filter(MaskedData[:InitializeNum])
    x_new = np.zeros((3, MaskedData.shape[0]))
    x_new[0,0:InitializeNum] = filtered_state_means[:,0]
    x_new[1,0:InitializeNum] = filtered_state_means[:,2]
    x_new[2,0:InitializeNum] = filtered_state_means[:,4]

    filtered_state_means = filtered_state_means[-1]
    filtered_state_covariances = filtered_state_covariances[-1]

    # import ipdb;ipdb.set_trace()
    ShiftCounter = 1
    for i in range(MaskedData.shape[0]):
        if i < InitializeNum:
            continue
        # # import ipdb;ipdb.set_trace()
        # PredictedStateNext = kf1.transition_matrices.dot(filtered_state_means)
        # PredictedNextPointList = [PredictedStateNext[0],PredictedStateNext[2],PredictedStateNext[4]]
        # Diff = EucDist(np.array(PredictedNextPointList),MaskedData[i])
        # # Diff = np.mean(np.array(PredictedNextPointList)-MaskedData[i])

        
        # if Diff > 100:
        #     # x_new[0,i] = PredictedNextPointList[0]
        #     # x_new[1,i] = PredictedNextPointList[1]
        #     # x_new[2,i] = PredictedNextPointList[2]
        #     x_new[0,i] = np.nan
        #     x_new[1,i] = np.nan
        #     x_new[2,i] = np.nan
        #     filtered_state_means = PredictedStateNext
        #     # filtered_state_covariances = kf1.transition_covariance + kf1.transition_matrices.dot(kf1.initial_state_covariance).dot(kf1.transition_matrices.T)
        #     # (filtered_state_means, filtered_state_covariances)  = kf1.filter_update(filtered_state_means, filtered_state_covariances, MaskedData[i-ShiftCounter])
        #     ShiftCounter += 1
        #     continue
        # else:
        (filtered_state_means, filtered_state_covariances)  = kf1.filter_update(filtered_state_means, filtered_state_covariances, MaskedData[i])
        x_new[0,i] = filtered_state_means[0]
        x_new[1,i] = filtered_state_means[2]
        x_new[2,i] = filtered_state_means[4]
        ShiftCounter = 1
    # kf1 = kf1.em(MaskedData, n_iter=5)
    # (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(MaskedData)
    # (smoothed_state_means, smoothed_state_covariances) = kf1.filter(MaskedData)


    # # import ipdb;ipdb.set_trace()
    # plt.figure(1)
    # times = range(MaskedData.shape[0])
    # plt.plot(times, MaskedData[:, 0], 'bo',
    #         times, MaskedData[:, 1], 'ro',
    #         times, MaskedData[:, 2], 'go',
    #         times, x_new[0,:], 'b--',
    #         times, x_new[1,:], 'r--',
    #         times, x_new[2, :], 'g--',
    #         markersize=1, linewidth=2)
    # plt.show()

    Results = [x_new[0,:],x_new[1,:],x_new[2,:]]

    return Results



def RunInterpolation3D(Predictions3D):

    NewPredictions = {}
    for key,val in Predictions3D.items():
        NewDict = {}

        for k,v in val.items():
            if type(v) == float: #if type is float, it is nan
                NewDict["%s_x"%k] = np.nan
                NewDict["%s_y"%k] = np.nan
                NewDict["%s_z"%k] = np.nan
            else:
                NewDict["%s_x"%k] = v[0]
                NewDict["%s_y"%k] = v[1]
                NewDict["%s_z"%k] = v[2]
        NewPredictions[key] = NewDict

    data = pd.DataFrame.from_dict(NewPredictions,orient= "index")
    # LinearInterpolate = data.interpolate(type="index", axis = 0)
    # SplineInterpolate = data.interpolate(type="spline",degree=5, axis = 0)

    ###Kalman
    KalmanData = data.copy()
    UnqNames = natsort.natsorted(list(set([col[:-2] for col in data.columns])))

    for name in tqdm(UnqNames):
        ColNames = ["%s_x"%name,"%s_y"%name,"%s_z"%name]
        ColVals = data[ColNames].to_numpy()
        # NewVals = RunKalman(ColVals)
        NewVals = RunRollingAverage(ColVals)
        KalmanData[ColNames[0]] = NewVals[:,0]
        KalmanData[ColNames[1]] = NewVals[:,1]
        KalmanData[ColNames[2]] = NewVals[:,2]


    # import ipdb;ipdb.set_trace()
    UnqNames = natsort.natsorted(list(set([col[:-2] for col in data.columns])))
    NewDF = pd.DataFrame(columns = UnqNames)

    for name in UnqNames:
        ColNames = ["%s_x"%name,"%s_y"%name,"%s_z"%name]
        ListVal = KalmanData[ColNames].values.tolist()
        ##If any of the dimensions are nan, make it all nan
        # ListVal2 = [val if not any(np.isnan(val)) else [np.nan,np.nan,np.nan] for val in ListVal ]
        ListVal2 = [val if not any(np.isnan(val)) else np.nan for val in ListVal]

        NewDF[name] = ListVal2


    NewDF = NewDF.applymap(np.array)
    NewDF.index = data.index
    FinalDict = NewDF.to_dict(orient="index")


    # import ipdb;ipdb.set_trace()
    TempDict = pd.DataFrame.from_dict(Predictions3D,orient= "index")
    NANBefore = np.sum(TempDict.isna().sum().to_numpy())
    NANAfter = np.sum(NewDF.isna().sum().to_numpy())

    # import ipdb;ipdb.set_trace()


    PercentageKPRemoved = (NANAfter-NANBefore)*100/NewDF.size
    print("Total Removed: %s%%"%((NANAfter-NANBefore)*100/NewDF.size))

    return FinalDict,PercentageKPRemoved


def applyKalman(EvalDir,Sequences,ModelName):
    #CamNames = ["Cam1","Cam2","Cam3","Cam4"]
    #PickleFiles = [os.path.basename(file) for file in glob(EvalDir + "/*.p") if "Points3D" in file or "Points2D" in file]
    #Files3D = [file for file in PickleFiles if "Points3D" in file]
    # import ipdb;ipdb.set_trace()
    PercentageDict = {ModelName:{}}
    # for file3d in tqdm(Files3D):
        # file3d = Files3D[101]

    # _, _, ModelName, SeqNum= file3d.split("_")
    # SeqNum = SeqNum.split(".")[0].split("Seq")[1]
    
    for SeqNum in tqdm(Sequences):
        print("Sequence: %s, Model: %s"%(SeqNum,ModelName))
        # import ipdb;ipdb.set_trace()
        # if os.path.exists(os.path.join(EvalDir,"SeqEval_Kalman3D_%s_Seq%s.p"%(ModelName,SeqNum))):
        #     continue
        Predictions3D= pickle.load(open(os.path.join(EvalDir,"SeqEval_Points3D_%s_Seq%s.p"%(ModelName,SeqNum)),"rb"))
        # Predictions2D = pickle.load(open(os.path.join(EvalDir,"SeqEval_Points2D_%s_Seq%s.p"%(ModelName,SeqNum) ), "rb"))
        KalmanOut,PercentageKPRemoved = RunInterpolation3D(Predictions3D)
        PercentageDict[ModelName][SeqNum] = PercentageKPRemoved
        # pickle.dump(LinearOut, open(os.path.join(EvalDir,"SeqEval_Linear3D_%s_Seq%s.p"%(ModelName,SeqNum)), "wb"))
        # pickle.dump(SplineOut, open(os.path.join(EvalDir,"SeqEval_Spline3D_%s_Seq%s.p"%(ModelName,SeqNum)), "wb"))
        pickle.dump(KalmanOut, open(os.path.join(EvalDir,"SeqEval_RollingFilter3D_%s_Seq%s.p"%(ModelName,SeqNum)), "wb"))

    FinalDF = pd.DataFrame.from_dict(PercentageDict)
    print("Mean % Removed:")
    print(FinalDF.apply(np.mean,axis = 0))



def ParseArgs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path",
                        type=str,
                        required=True,
                        help="path with inference files")
    parser.add_argument("--name",
                        type=str,
                        required=True,
                        help="Name for the whole framework")
    parser.add_argument("--Sequences",
                        type=list,
                        required=True,
                        help="Sequences to apply the kalman filter to")

    arg = parser.parse_args()

    return arg

if __name__ == "__main__":
    args = ParseArgs()
    EvalDir = args.path
    ModelName = args.name
    Sequences = args.Sequences
    applyKalman(EvalDir,Sequences,ModelName)



    # import ipdb;ipdb.set_trace()
