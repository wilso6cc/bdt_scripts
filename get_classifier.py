import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from scipy.stats import loguniform, randint
from sklearn.metrics import roc_curve, roc_auc_score

path = 'Data/'

def get_classifier(dataframe):

    features = dataframe.drop(['Signal', 'Weight'], axis=1)
    feature_names = features.columns.tolist()
    
    target = dataframe['Signal']
    weights = dataframe['Weight']
    
    x_train, x_test, y_train, y_test, weight_train, weight_test = train_test_split(features, target, weights, train_size=0.8)
    bdt = HistGradientBoostingClassifier()
    classifier = bdt.fit(x_train, y_train, sample_weight=weight_train)

    return classifier, feature_names

def truncated_df(uproot_tree, keys_needed):
    return uproot_tree.arrays(keys_needed, library='pd')

ww_tree = uproot.open(path+"WW_MG5_NLO_wID.root:OS_MuE_Reco")
dfdy_tree = uproot.open(path+"DFDY_Pythia_LO_wID.root:OS_MuE_Reco")
kinematic_keys = ["_PT","_ETA","_PHI","DeltaRLL","DeltaPhiLL","DeltaEtaLL","DiLeptonpT","DiLeptonMass","Weight"]

list_of_keys_needed = []
for key1 in ww_tree.keys():
    for name in kinematic_keys:
        if name in key1:
            list_of_keys_needed.append(key1)

ww_df = truncated_df(ww_tree, list_of_keys_needed)
ww_df['Signal'] = 1
dfdy_df = truncated_df(dfdy_tree, list_of_keys_needed)
dfdy_df['Signal'] = 0

df_tot = pd.concat([ww_df, dfdy_df])
trained_model, feature_names = get_classifier(df_tot)

trained_bdt_info = {
    'Model': trained_model,
    'Features': feature_names,
    'Dataframe': df_tot
}

from pickle import dump
with open("WWDFDY_trained_bdt.pkl", "wb") as f:
    dump(trained_bdt_info, f, protocol=5)


print('\nBDT Exported as pickle successfully!')