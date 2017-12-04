import json
import cPickle as pickle
import numpy as np

from bubbly.model import Model, ModelGroup
from bubbly.extractors import MultiViewExtractor, ManyManyExtractors
from bubbly.dr1 import WideLocationGenerator,LocationGenerator
from bubbly.wiserf import WiseRF
#from sklearn.ensemble import RandomForestClassifier

def add_traningset_1(data,lon):
    for ctt_l in range(10):
            for ctt_b in range(4):
                data['pos'].append([lon, lon%360-0.95+ctt_l*0.1, (ctt_b-1)*0.1, 0.046])
    return data
            
def add_traningset_2(data,lon):
    for ctt_l in range(10):
            for ctt_b in range(4):
                data['pos'].append([lon, lon%360-0.95+ctt_l*0.1, (ctt_b-1)*0.1, 0.038])
    return data
    
def add_traningset_neg(data,lon):
    for ctt_l in range(20):
        for ctt_b in range(8):
            data['neg'].append([lon, lon%360-0.95+ctt_l*0.1, (ctt_b-3.5)*0.1, 0.046])     
    return data
    
                
def make_model(mod3):
    params = {'max_features': 'auto',
              'n_jobs': 2,
              'min_samples_split': 4,
#              'criterion': 'infogain',
              'criterion': 'gini',    ###  entropy
#              'criterion': 'entropy',    ###  
              'n_estimators': 800}
    ex = MultiViewExtractor(ManyManyExtractors())
    loc = WideLocationGenerator(mod3)
#    clf = RandomForestClassifier(**params)
    clf = WiseRF(**params)
    return Model(ex, loc, clf)


def train_model(model, mod3):
#    data = json.load(open('../models/training_data_%i.json' % mod3))
    data = json.load(open('../models/training_dataxdno_%i.json' % mod3))
#    data = json.load(open('../models/training_dataxd_%i.json' % mod3))


#    if mod3==0:
#        for lon_all in np.array([71,82,74,76])+360:
#            data=add_traningset_1(data,np.int(lon_all))
#        for lon_all in np.array([73,85,77])+360:
#            data=add_traningset_2(data,np.int(lon_all))
#        data=add_traningset_neg(data,82)
#    if mod3==1:
#        for lon_all in np.array([71,72,74,86])+360:
#            data=add_traningset_1(data,np.int(lon_all))
#        for lon_all in np.array([83,75,77])+360:
#            data=add_traningset_2(data,np.int(lon_all))    
#        data=add_traningset_neg(data,83)
#    if mod3==2:
#        for lon_all in np.array([81,72,84,76])+360:
#            data=add_traningset_1(data,np.int(lon_all))
#        for lon_all in np.array([73,75,87])+360:
#            data=add_traningset_2(data,np.int(lon_all))  
#        data=add_traningset_neg(data,82)
        
    if mod3==0:
        for lon_all in np.array([71,82,74,76,121,112,124,116])+360:
#        for lon_all in np.array([121,112,124,116])+360:
            data=add_traningset_1(data,np.int(lon_all))
        for lon_all in np.array([73,85,77,113,115,127])+360:
#        for lon_all in np.array([113,115,127])+360:
            data=add_traningset_2(data,np.int(lon_all))
        data=add_traningset_neg(data,82)
    if mod3==1:
        for lon_all in np.array([71,72,74,86,111,122,114,116])+360:
#        for lon_all in np.array([111,122,114,116])+360:
            data=add_traningset_1(data,np.int(lon_all))
        for lon_all in np.array([83,75,77,113,125,117])+360:
#        for lon_all in np.array([113,125,117])+360:
            data=add_traningset_2(data,np.int(lon_all))    
        data=add_traningset_neg(data,83)
    if mod3==2:
        for lon_all in np.array([81,72,84,76,111,112,114,126])+360:
#        for lon_all in np.array([111,112,114,126])+360:
            data=add_traningset_1(data,np.int(lon_all))
        for lon_all in np.array([73,75,87,123,115,117])+360:
#        for lon_all in np.array([123,115,117])+360:
            data=add_traningset_2(data,np.int(lon_all))  
        data=add_traningset_neg(data,82)
         
    
    model.fit(data['pos'], data['neg'])
    return model

#"""
def main():

    models = [train_model(make_model(i), i) for i in [0, 1, 2]]
#    models = [train_model(make_model(i), i) for i in [0]]
    mg = ModelGroup(*models)
#    mg.save('../models/full_classifier_retrain_xd_all_0417_noise.dat')
#    mg.save('../models/full_classifier_retrain_xd_all_entropy_0430.dat')
#    mg.save('../models/full_classifier_retrain_xd_all_gini_0528.dat')
#    mg.save('../models/full_classifier_xd_entropy_0528.dat')
#    mg.save('../models/full_classifier_xd_reduceMWP_simulation_1025.dat')
#    mg.save('../models/full_classifier_xd_only_simulation_1029.dat')
#    mg.save('../models/full_classifier_xd_retrain_noise_1030.dat')
    mg.save('../models/full_classifier_xd_only_sim_non_noi_1102.dat')
#    mg.save('../models/full_classifier_retrain_xd_all_gini_0528.dat')
#    mg.save('../models/full_classifier_xd_all_entropy_0430.dat')
#    mg.save('../models/full_classifier_xd_all_0417.dat')

if __name__ == "__main__":
    main()

