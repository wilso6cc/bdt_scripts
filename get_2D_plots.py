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
df_tot = imported_trained_bdt['Dataframe']

def run_bdt(classifier, new_df, feature_names):

    if 'Scale Factor' in new_df.keys():
        new_df['Weight'] = new_df['Scale Factor'] * new_df['Reweight']
    x_input = new_df[feature_names]
    
    probabilities = classifier.predict_proba(x_input)
    bdt_scores = probabilities[:, 1]

    results_df = new_df.copy()
    results_df['bdt_score'] = bdt_scores
    
    return results_df

def plotting(df_tot, new_process, new_process_name):
    cut_val=0.36
    WW = df_tot[df_tot['Signal']==1]
    Zττ = df_tot[df_tot['Signal']==0]

    x_vars_list = ['DiLeptonMass','DiLeptonMass', 'DiLeptonMass']
    y_vars_list = ['DeltaPhiLL','DeltaEtaLL','DeltaRLL']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
    for i, (x_var, y_var) in enumerate(zip(x_vars_list, y_vars_list)):

        ax_og = axes[i, 0]
        ax_new = axes[i, 1]

        ax_og.scatter(WW[x_var], WW[y_var], color='dodgerblue', s=10, alpha=0.1)
        ax_og.scatter([], [], color='dodgerblue', label=r'$W^+W^-$')

        ax_og.scatter(Zττ[x_var], Zττ[y_var], color='red', s=10, alpha=0.1)
        ax_og.scatter([], [], color='red', label=r'Z $\rightarrow \tau\tau$')

        ax_og.set_xlabel(x_var, fontsize=16)
        ax_og.set_ylabel(y_var, fontsize=20)
        ax_og.set_title(r'WW and Z$\rightarrow$ττ', fontsize=34, pad=15, fontweight='bold')
        ax_og.legend(fontsize=16)
        ax_og.grid(True, linestyle='--', alpha=0.5)

        ax_new.scatter(new_process[x_var], new_process[y_var], color='silver', s=10, alpha=0.25)
        ax_new.scatter([], [], color='silver', label=new_process_name)

        ax_new.set_xlabel(x_var, fontsize=16)
        ax_new.set_ylabel(y_var, fontsize=20)
        ax_new.set_title(f'{new_process_name} Distribution', fontsize=34, pad=15, fontweight='bold')
        ax_new.legend(fontsize=16)
        ax_new.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig.savefig(f"Process_Distributions/{new_process_name}_distribution.png")

files_to_skip = ["WW_MG5_NLO_wID.root", "DFDY_Pythia_LO_wID.root"]

for file_path in path_to_data_folder.rglob('*wID.root'): 
    if file_path.name not in files_to_skip:
        new_tree = uproot.open(path_to_data_folder.name + '/' + file_path.name + ":OS_MuE_Reco")
        actual_keys = new_tree.keys()

        ww_tree = uproot.open(path_to_data_folder.name + "/WW_MG5_NLO_wID.root:OS_MuE_Reco")
        dfdy_tree = uproot.open(path_to_data_folder.name + "/DFDY_Pythia_LO_wID.root:OS_MuE_Reco")
        kinematic_keys = ["_PT","_ETA","_PHI","DeltaRLL","DeltaPhiLL","DeltaEtaLL","DiLeptonpT","DiLeptonMass"]

        list_of_keys_needed = [k for k in actual_keys if any(name in k for name in kinematic_keys)]

        norm_weight_column = 'Weight' in actual_keys
        sfactor_reweight_columns = 'Scale Factor' in actual_keys and 'Reweight' in actual_keys

        if norm_weight_column:
            list_of_keys_needed.append('Weight')
        elif sfactor_reweight_columns:
            list_of_keys_needed.extend(['Scale Factor', 'Reweight'])

        def truncated_df(uproot_tree, keys_needed):
            return uproot_tree.arrays(keys_needed, library='pd')

        new_process_df = truncated_df(new_tree, list_of_keys_needed)
        new_process_df['Signal'] = 0

        list_of_processes = ['ttbar','DFDY','tW','WZ','ZZ']
        for process in list_of_processes:
            if process in file_path.name:
                new_process_name = process

        results = run_bdt(bdt_model, new_process_df, required_features)

        plotting(df_tot, new_process_df, new_process_name)
        new_process_name = []

exit()