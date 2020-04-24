#!/usr/bin/env python

import os
import numpy as np
from scipy import signal
from keras import backend as K
from keras.models import load_model


def run_12ECG_classifier(data,header_data,classes,M):

    num_classes = len(classes)

    #preProcessing data
    data = data.T
    data = signal.resample(data, int(data.shape[0]*256/500), axis=0)
    x = np.zeros((1,15360, 12))
    if data.shape[0]>15360:
        data = data[:15360,:]
    x[:,:data.shape[0],:]=data

    # Use your classifier here to obtain a label and score for each class.
    models = M[0]
    Emodels = M[1]
    '''
    prediction = np.zeros((len(models), num_classes))
    for i in range(len(models)):
        prediction[i, :] = models[i].predict(x).copy()
    current_score = np.mean(prediction, axis=0)
    current_label = np.where(current_score > 0.5, 1, 0)
    '''
    prediction = np.zeros((len(Emodels), num_classes))
    for i in range(len(Emodels)):
        lead_models = models[i*13:(i+1)*13]
        for lead in range(13):
            tmpx = x.copy()
            if lead!=12:
                zeroIndices = np.asarray(list(set(range(12)) - set([lead])))
                tmpx[:,:,zeroIndices] = 0
            if lead==0:
                lead_prediction = lead_models[lead].predict(tmpx).copy()
            else:
                lead_prediction = np.concatenate((lead_prediction, lead_models[lead].predict(tmpx).copy()),axis=1)
        lead_prediction = np.expand_dims(lead_prediction, axis=-1)
        prediction[i] = Emodels[i].predict(lead_prediction).copy()
    
    current_score = np.mean(prediction, axis=0)
    current_label = np.where(current_score > 0.5, 1, 0)
    
    return current_label, current_score


def load_12ECG_model():
    # load the model from disk
    import net
    '''
    filename = ['10_fold_model_'+str(fold+1)+'.hdf5' for fold in range(10)]
    models = []
    for modelname in filename:
        model = load_model(modelname, {'GeometricMean': GeometricMean})
        models.append(model)
    '''
    leads_name = ['I','II','III','avR','avL','avF','V1','V2','V3','V4','V5','V6']
    model_path = 'models/'
    models = []
    Emodels = []
    M = []
    count = 0
    for fold in range(10):
        if fold!=6:
            continue
        for lead in range(13):
            count += 1
            print('loading %d/14 model...'%count)
            model = net.Net()
            if lead==12:
                model.load_weights(model_path+'10_fold_model_'+str(fold+1)+'.hdf5')
            else:
                model.load_weights(model_path+'10_fold_model_'+leads_name[lead]+'_'+str(fold+1)+'.hdf5')
            models.append(model)
        
        Emodel = net.ensemble_model() 
        Emodel.load_weights(model_path+'10_fold_Emodel_'+str(fold+1)+'.hdf5')
        Emodels.append(Emodel)
    M.append(models)
    M.append(Emodels)
    return M
