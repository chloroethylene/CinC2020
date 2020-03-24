#!/usr/bin/env python

import os
import numpy as np
import net
from get_12ECG_features import get_12ECG_features

def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    #current_label = np.zeros(num_classes, dtype=int)
    #current_score = np.zeros(num_classes)

    #fix data
    data = data.T
    fix_data = np.zeros((1,15360, 12))
    if data.shape[0]>15360:
        data = data[:15360,:]
    fix_data[:,:data.shape[0],:]=data

    # Use your classifier here to obtain a label and score for each class. 
    #features=np.asarray(get_12ECG_features(data,header_data))
    #feats_reshape = features.reshape(1,-1)
    #label = model.predict(feats_reshape)
    
    prediction = np.zeros((len(model), num_classes))
    for i in range(len(model)):
        prediction[i, :] = model[i].predict(fix_data).copy()
    current_score = np.mean(prediction, axis=0)

    current_label = np.where(current_score > 0.5, 1, 0)

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    filename = ['model_fold_'+str(fold+1)+'.h5' for fold in range(4)]
    loaded_model = []
    model = net.Net(9, 15360, 12, None)
    for modelname in filename:
        model.load_weights(modelname)
        loaded_model.append(model)
    return loaded_model
