import os
import sys

import math
from itertools import product

import numpy
import pandas

import lightgbm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 

from scipy.stats import pearsonr

import shap

import gridcv

'''
Vars
'''
rootPath = ""
dataPath = ""
targetPath = ""
savePath = ""

normalize = True # not necessary for lgbm but do it for interpretability 
dropna = False # not rlly necessary for lgbm, built in handler for nan rows
testSetSize = 0.2
validSetSize = 0.2

validationR2 = pandas.DataFrame(columns = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
validationPearsonR = pandas.DataFrame(columns = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])

testR2 = pandas.DataFrame(columns = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
testPearsonR = pandas.DataFrame(columns = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])

'''
Data Loading
'''
input = pandas.read_csv(dataPath)
targets = pandas.read_csv(targetPath)

'''
Data Preprocessing
'''

# normalize y features
# drop nan rows

if normalize:
    s = MinMaxScaler(feature_range=(0,1))
    targets = pandas.DataFrame(s.fit_transform(targets), columns=targets.columns, index=targets.index)
    print("normalized targets between 0,1")

if dropna:
    input = input.dropna(0)
    print("dropped nan rows")

'''
Data Splitting 
'''
xTrain, xTest, yTrain, yTest = train_test_split(input, 
                                                    targets, 
                                                    test_size=testSetSize, 
                                                    shuffle=False) #shuffle set to false to prevent temporal leakage


'''
5 fold CV splitting
'''
# in Liu et al., 2024: for each subject and session 5 models trained and assessed for each fold with shap values extracted
#                      hyperparamter tuning is also performed for each fold on the train/valid

indicesPerWindow = math.ceil(xTrain.shape[0] / 9)
indices = [i * indicesPerWindow for i in range(9 + 1)]
indices[len(indices) - 1] = xTrain.shape[0]

fold1XTrain = xTrain.iloc[indices[0]:indices[4], :]
fold1YTrain = yTrain.iloc[indices[0]:indices[4], :]
fold1XTest = xTrain.iloc[indices[4]:indices[5], :]
fold1YTest = yTrain.iloc[indices[4]:indices[5], :]

fold2XTrain = xTrain.iloc[indices[1]:indices[5], :]
fold2YTrain = yTrain.iloc[indices[1]:indices[5], :]
fold2XTest = xTrain.iloc[indices[5]:indices[6], :]
fold2YTest = yTrain.iloc[indices[5]:indices[6], :]

fold3XTrain = xTrain.iloc[indices[2]:indices[6], :]
fold3YTrain = yTrain.iloc[indices[2]:indices[6], :]
fold3XTest = xTrain.iloc[indices[6]:indices[7], :]
fold3YTest = yTrain.iloc[indices[6]:indices[7], :]

fold4XTrain = xTrain.iloc[indices[3]:indices[7], :]
fold4YTrain = yTrain.iloc[indices[3]:indices[7], :]
fold4XTest = xTrain.iloc[indices[7]:indices[8], :]
fold4YTest = yTrain.iloc[indices[7]:indices[8], :]

fold5XTrain = xTrain.iloc[indices[4]:indices[8], :]
fold5YTrain = yTrain.iloc[indices[4]:indices[8], :]
fold5XTest = xTrain.iloc[indices[8]:indices[9], :]
fold5YTest = yTrain.iloc[indices[8]:indices[9], :]

'''
Model Training
'''
# There exists more robust hyperparameter tuning methods (bayesian etc.) Grid search is just fine for our purposes.
paramGrid = {
    'num_leaves': [3, 5, 13, 29],
    'bagging_fraction': [ 0.7, 0.8, 0.9, 1.0],
    'bagging_freq': [0, 4, 8, 12],
    'feature_fraction': [0.7, 0.8, 0.9, 1.0],
    'learning_rate': numpy.logspace(-3, -1, 5),
}

combs = list(product(
    paramGrid['num_leaves'],
    paramGrid['bagging_fraction'],
    paramGrid['bagging_freq'],
    paramGrid['feature_fraction'],
    paramGrid['learning_rate']
))
print("tuning hyperparameters")
bestParamsFold1, bestMSEFold1 = gridcv.tuneLGBRegressionParameters(combs, fold1XTrain, fold1YTrain, fold1XTest, fold1YTest)
bestParamsFold2, bestMSEFold2 = gridcv.tuneLGBRegressionParameters(combs, fold2XTrain, fold2YTrain, fold2XTest, fold2YTest)
bestParamsFold3, bestMSEFold3 = gridcv.tuneLGBRegressionParameters(combs, fold3XTrain, fold3YTrain, fold3XTest, fold3YTest)
bestParamsFold4, bestMSEFold4 = gridcv.tuneLGBRegressionParameters(combs, fold4XTrain, fold4YTrain, fold4XTest, fold4YTest)
bestParamsFold5, bestMSEFold5 = gridcv.tuneLGBRegressionParameters(combs, fold5XTrain, fold5YTrain, fold5XTest, fold5YTest)

fold1GBMData = lightgbm.Dataset(fold1XTrain, label=fold1YTrain)
fold1Model = lightgbm.train(bestParamsFold1, fold1GBMData)
fold2GBMData = lightgbm.Dataset(fold2XTrain, label=fold2YTrain)
fold2Model = lightgbm.train(bestParamsFold2, fold2GBMData)
fold3GBMData = lightgbm.Dataset(fold3XTrain, label=fold3YTrain)
fold3Model = lightgbm.train(bestParamsFold3, fold3GBMData)
fold4GBMData = lightgbm.Dataset(fold4XTrain, label=fold4YTrain)
fold4Model = lightgbm.train(bestParamsFold4, fold4GBMData)
fold5GBMData = lightgbm.Dataset(fold5XTrain, label=fold5YTrain)
fold5Model = lightgbm.train(bestParamsFold5, fold5GBMData)

'''
Model testing

'''
fold1Preds = fold1Model.predict(fold1XTest)
fold2Preds = fold2Model.predict(fold2XTest)
fold3Preds = fold3Model.predict(fold3XTest)
fold4Preds = fold4Model.predict(fold4XTest)
fold5Preds = fold5Model.predict(fold5XTest)


'''
Model evaluation
'''
fold1R2 = r2_score(fold1YTest, fold1Preds)
fold2R2 = r2_score(fold2YTest, fold2Preds)
fold3R2 = r2_score(fold3YTest, fold3Preds)
fold4R2 = r2_score(fold4YTest, fold4Preds)
fold5R2 = r2_score(fold5YTest, fold5Preds)

fold1PearsonR, _ = pearsonr(fold1YTest.values.flatten(), fold1Preds)
fold2PearsonR, _ = pearsonr(fold2YTest.values.flatten(), fold2Preds)
fold3PearsonR, _ = pearsonr(fold3YTest.values.flatten(), fold3Preds)
fold4PearsonR, _ = pearsonr(fold4YTest.values.flatten(), fold4Preds)
fold5PearsonR, _ = pearsonr(fold5YTest.values.flatten(), fold5Preds)

'''
Pull Shap values
'''

explainer1 = shap.TreeExplainer(fold1Model)
shapValues1 = explainer1(fold1XTest)
explainer2 = shap.TreeExplainer(fold2Model)
shapValues2 = explainer2(fold2XTest)
explainer3 = shap.TreeExplainer(fold3Model)
shapValues3 = explainer3(fold3XTest)
explainer4 = shap.TreeExplainer(fold4Model)
shapValues4 = explainer4(fold4XTest)
explainer5 = shap.TreeExplainer(fold5Model)
shapValues5 = explainer5(fold5XTest)

'''
IO
'''
validationR2.loc[0] = [fold1R2, fold2R2, fold3R2, fold4R2, fold5R2]
validationR2.to_csv(os.path.join(savePath, "validationR2.csv"), index=False)

validationPearsonR.loc[0] = [fold1PearsonR, fold2PearsonR, fold3PearsonR, fold4PearsonR, fold5PearsonR]
validationPearsonR.to_csv(os.path.join(savePath, "validationPearsonR.csv"), index=False)

shapValues1 = pandas.DataFrame(shapValues1.values, columns = fold1XTest.columns)
shapValues1.to_csv(os.path.join(savePath, "shapValuesFold1.csv"), index=False)
shapValues2 = pandas.DataFrame(shapValues2.values, columns = fold2XTest.columns)
shapValues2.to_csv(os.path.join(savePath, "shapValuesFold2.csv"), index=False)
shapValues3 = pandas.DataFrame(shapValues3.values, columns = fold3XTest.columns)
shapValues3.to_csv(os.path.join(savePath, "shapValuesFold3.csv"), index=False)
shapValues4 = pandas.DataFrame(shapValues4.values, columns = fold4XTest.columns)
shapValues4.to_csv(os.path.join(savePath, "shapValuesFold4.csv"), index=False)
shapValues5 = pandas.DataFrame(shapValues5.values, columns = fold5XTest.columns)
shapValues5.to_csv(os.path.join(savePath, "shapValuesFold5.csv"), index=False)

fold1Model.save_model(os.path.join(savePath, "fold1Model.txt"))
fold2Model.save_model(os.path.join(savePath, "fold2Model.txt"))
fold3Model.save_model(os.path.join(savePath, "fold3Model.txt"))
fold4Model.save_model(os.path.join(savePath, "fold4Model.txt"))
fold5Model.save_model(os.path.join(savePath, "fold5Model.txt"))

'''
Evaluate on holdout set

Have to look at fold results and pick best model params
'''

holdoutParams = bestParamsFold3 # change this
holdoutGBMData = lightgbm.Dataset(xTrain, label=yTrain)
holdoutModel = lightgbm.train(holdoutParams, holdoutGBMData)
holdoutPreds = holdoutModel.predict(xTest)

holdoutR2 = r2_score(yTest, holdoutPreds)
holdoutPearsonR, _ = pearsonr(yTest.values.flatten(), holdoutPreds)

