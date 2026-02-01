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
import sys
from pathlib import Path
import csv
import subprocess

# Provide the path to a folder where all of the data for different processes is stored.
path_to_data_folder = Path(input('\nPath to Folder with Data: '))
# pathlib_call = Path(path_to_data_folder)

if sys.argv[0].endswith('.py'):
    print(f'Running {sys.argv[1]} on process in {path_to_data_folder.name}...\n')
else:
    print('\nError: Use the following format: python3 run_bdt.py <pickle file>\n')
    exit()

from pickle import load
with open(sys.argv[1], "rb") as f:
    imported_trained_bdt = load(f)

bdt_model = imported_trained_bdt["Model"]
required_features = imported_trained_bdt["Features"]

def run_bdt(classifier, new_df, feature_names):

    if 'Scale Factor' in new_df.keys():
        new_df['Weight'] = new_df['Scale Factor'] * new_df['Reweight']
    x_input = new_df[feature_names]
    
    probabilities = classifier.predict_proba(x_input)
    bdt_scores = probabilities[:, 1]

    results_df = new_df.copy()
    results_df['bdt_score'] = bdt_scores
    
    return results_df

process_passed_scores, list_of_processes = [], []
bdt_prob_cutoff = 0.36

for file_path in path_to_data_folder.rglob('*wID.root'): 
    new_tree = uproot.open(path_to_data_folder.name + '/' + file_path.name + ":OS_MuE_Reco")

    ww_tree = uproot.open(path_to_data_folder.name + "/WW_MG5_NLO_wID.root:OS_MuE_Reco")
    dfdy_tree = uproot.open(path_to_data_folder.name + "/DFDY_Pythia_LO_wID.root:OS_MuE_Reco")
    kinematic_keys = ["_PT","_ETA","_PHI","DeltaRLL","DeltaPhiLL","DeltaEtaLL","DiLeptonpT","DiLeptonMass","Weight"]

    list_of_keys_needed = []
    for key1 in ww_tree.keys():
        for name in kinematic_keys:
            if name in key1:
                list_of_keys_needed.append(key1)

    SF_and_RW_keys = list_of_keys_needed.extend(['Scale Factor', 'Reweight'])

    def truncated_df(uproot_tree, keys_needed):
        return uproot_tree.arrays(keys_needed, library='pd')

    new_process_df = truncated_df(new_tree, SF_and_RW_keys)
    new_process_df['Signal'] = 0

    results = run_bdt(bdt_model, new_process_df, required_features)
    
    passed_events, failed_events = 0, 0
    
    for score in results['bdt_score']:
        if score > bdt_prob_cutoff:
            passed_events = passed_events+1
        else:
            failed_events = failed_events+1

    total_events = passed_events + failed_events
    # print(f"\nEvents that passed BDT: {passed_events} \n% of Total: {passed_events/total_events}")
    # print(f"Events that failed BDT: {failed_events} \n% of Total: {failed_events/total_events}\n")

    efficiency = passed_events/total_events
    process_passed_scores.append(efficiency)
    list_of_processes.append(file_path.name)

output_table = [
    ['Process',  'BDT Efficiency']
]

for process, score in zip(list_of_processes, process_passed_scores):
    output_table.append([process, score])

output_table_path = input("What would you like to call the output table: ") + '.csv'

with open(output_table_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(output_table)

print(f"Table successfully written to {output_table_path}. Exiting...\n")

subprocess.call(['open', output_table_path])