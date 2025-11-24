import numpy

import lightgbm

from sklearn.metrics import mean_squared_error, log_loss
def tuneLGBRegressionParameters(paramSet, xTrain, yTrain, xTest, yTest):
    
    bestParam = None
    bestMSE = numpy.float('inf')
    for (numLeaves, bagFrac, bagFreq, featFrac, learnRate) in paramSet:
        params = {
			"objective": 'regression',
			"boosting_type": 'gbdt',
			"force_col_wise": True,
			"metric": 'l2',
			"n_estimators": 1000,
            'num_leaves': numLeaves,
            'bagging_fraction': bagFrac,
            'bagging_freq': bagFreq,
            'feature_fraction': featFrac,
            'learning_rate': learnRate,
            'random_state': 42 # reproducibility
        }
        dataSet = lightgbm.Dataset(xTrain, label=yTrain)
        model = lightgbm.train(params, dataSet)
        preds = model.predict(xTest)
        mse = mean_squared_error(yTest, preds)

        print("Params:", params, "MSE:", mse)

        if bestMSE is None or mse < bestMSE:
            bestMSE = mse
            bestParam = params
        
    return bestParam, bestMSE


def tuneLGBClassifierBinaryParameters(paramSet, xTrain, yTrain, xTest, yTest):
    
    bestParam = None
    bestLogLoss = numpy.float('inf')

    classes = numpy.unique(yTrain)
    classCounts = numpy.bincount(yTrain.values.flatten().astype(int))

    majorityClass = classCounts.max()
    minorityClass = classCounts.min()

    scalePosWeight = majorityClass / minorityClass

    for (numLeaves, bagFrac, bagFreq, featFrac, learnRate) in paramSet:
        params = {
			"objective": 'binary',
			"boosting_type": 'gbdt',
			"force_col_wise": True,
			"metric": 'binary_logloss',
			"n_estimators": 1000,
            'num_leaves': numLeaves,
            'bagging_fraction': bagFrac,
            'bagging_freq': bagFreq,
            'feature_fraction': featFrac,
            'learning_rate': learnRate,
            'random_state': 42, # reproducibility
            # if classes are imblaanced, tune for scale_pos_weight
            'scale_pos_weight': scalePosWeight
        }
        dataSet = lightgbm.Dataset(xTrain, label=yTrain)
        model = lightgbm.train(params, dataSet)
        preds = model.predict(xTest)
        logLoss = log_loss(yTest, preds)

        print("Params:", params, "Log Loss:", logLoss)

        if bestLogLoss is None or logLoss < bestLogLoss:
            bestLogLoss = logLoss
            bestParam = params
        
    return bestParam, bestLogLoss


