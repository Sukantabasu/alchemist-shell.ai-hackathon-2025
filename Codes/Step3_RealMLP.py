# Copyright (C) 2025 Sukanta Basu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
File: Step3_RealMLP.py
==========================

:Author: Sukanta Basu
:Date: 2025-8-14
:Description: ML predictions using the RealMLP model

Overall strategy:
Step 1: preprocessing and feature generation
Step 2: Use AutoGluon to generate OOF predictions for each target separately.
These predictions will be used as additional input features in steps 3 and 4.
Step 3: Train the RealMLP model with processed input (step 1) + ten
AutoGluon-OOFs (step 2). These additional features will capture the correlation
among targets effectively.
Step 4: Similar to step 3 except use the TabPFN (v2) model.
Step 5: Combine predictions from RealMLP (step 3) and TabPFN (step 4).

AI Assistance: Claude.AI (Anthropic) is used for documentation, code
restructuring, and performance optimization.
"""

# ============================================
# Imports
# ============================================

import numpy as np
import pandas as pd
import os
import random
import pickle

from scipy.stats import hmean
from sklearn.metrics import mean_absolute_percentage_error as mape

from pytabkit import RealMLP_TD_Regressor

# Set random seed for reproducibility
random.seed(7)
np.random.seed(7)

# Force numpy to use legacy RandomState instead of Generator
np.random.set_state(np.random.RandomState(7).get_state())


# ============================================
# User Input
# ============================================

# n-repetitions
nTrials = 100

# Number of folds in k-fold
nFolds = 8

# Number of input features + 10 OOFs
nFeatures = 65 + 10

# Number of target variables
nTargets = 10


# ============================================
# Input & Output Directories
# ============================================

ROOT_DIR = '/data/Sukanta/Works_AIML/2025_SHELL_FuelProperty/'
DATA_DIR = ROOT_DIR + 'DATA/'
ExtractedDATA_DIR = ROOT_DIR + 'ExtractedDATA/'
Tuning_DIR = ROOT_DIR + 'Models/RealMLP/'

# Create directory if it doesn't exist
os.makedirs(Tuning_DIR, exist_ok=True)

# ============================================
# Load Processed Training and Testing Data
# ============================================

df_XyTrnVal_org = pd.read_csv(ExtractedDATA_DIR + 'train_processed.csv')
nSamples_TrnVal = df_XyTrnVal_org.shape[0]

df_XTst = pd.read_csv(ExtractedDATA_DIR + 'test_processed.csv')
nSamples_Tst = df_XTst.shape[0]


# ============================================
# Load AutoGluon-generated OOF Data
# ============================================

df_XTrnVal_AG_OOF = pd.read_csv(ExtractedDATA_DIR + 'AutoGluon_21600_OOF.csv')
df_XTst_AG_OOF = pd.read_csv(ExtractedDATA_DIR + 'AutoGluon_21600_Tst.csv')


# ============================================
# Combine Dataframes
# ============================================

df_XyTrnVal = pd.concat([df_XTrnVal_AG_OOF, df_XyTrnVal_org], axis=1)
df_XTst = pd.concat([df_XTst_AG_OOF, df_XTst], axis=1)


# ============================================
# Initialize Storage for Results
# ============================================

dict_yTrnVal_OOF = {}
dict_yTst_pred_allFold = {}
dict_CV_scores = {}
dict_trained_models = {}

for trial in range(nTrials):
    dict_yTrnVal_OOF[trial] = {}
    dict_yTst_pred_allFold[trial] = {}
    dict_CV_scores[trial] = {}
    dict_trained_models[trial] = {}

# ============================================
# Iterative Single-target Training using RealMLP
# ============================================

nSamples_per_fold = int(nSamples_TrnVal / nFolds)

# n-repetitions of TabPFN models (resampling)
for trial in range(nTrials):

    print(f"\n=== TRIAL {trial + 1}/{nTrials} ===")

    # Shuffle training dataset & track original index
    shuffle_indx = np.random.permutation(nSamples_TrnVal)
    restore_indx = np.argsort(shuffle_indx)
    df_XyTrnVal_shuffled = (
        df_XyTrnVal.iloc[shuffle_indx].reset_index(drop=True))

    # Extract input features
    XTrnVal_shuffled = df_XyTrnVal_shuffled.iloc[:, 0:nFeatures].values

    # Multioutput targets
    for target in range(nTargets):

        print(f"\n--- Target {target + 1}/{nTargets} ---")

        # Extract single target from possible nTargets
        yTrnVal_shuffled = (
            df_XyTrnVal_shuffled.iloc[:, nFeatures + target].values)

        # Initialize zero vectors for OOF & test predictions
        yTrnVal_shuffled_pred = np.zeros_like(yTrnVal_shuffled)
        yTst_pred = np.zeros((nSamples_Tst, nFolds))

        # Store models for this target and trial
        dict_trained_models[trial][target] = []

        # K-folds
        for Fold in range(nFolds):
            # Create validation indices for this fold
            val_start = Fold * nSamples_per_fold
            val_end = min((Fold + 1) * nSamples_per_fold, nSamples_TrnVal)
            val_indices = list(range(val_start, val_end))

            # Create training indices (all except validation fold)
            trn_indices = list(range(0, val_start)) + list(
                range(val_end, nSamples_TrnVal))

            # Split features and targets
            XTrn_shuffled_fold = XTrnVal_shuffled[trn_indices]
            XVal_shuffled_fold = XTrnVal_shuffled[val_indices]

            yTrn_shuffled_fold = yTrnVal_shuffled[trn_indices]
            yVal_shuffled_fold = yTrnVal_shuffled[val_indices]

            print(
                f"  Fold {Fold + 1}/{nFolds}: "
                f"Train={len(trn_indices)}, "
                f"Val={len(val_indices)}")

            # Initialize RealMLP model
            regressor = RealMLP_TD_Regressor()

            # Fit (no tuning) using TabPFN model
            regressor.fit(XTrn_shuffled_fold, yTrn_shuffled_fold)

            # Store the trained model
            dict_trained_models[trial][target].append(regressor)

            # Make predictions on the holdout set
            yVal_shuffled_fold_pred = regressor.predict(XVal_shuffled_fold)
            yTrnVal_shuffled_pred[val_indices] = yVal_shuffled_fold_pred

            # Make predictions on the test set
            yTst_pred[:, Fold] = regressor.predict(df_XTst.iloc[:, 0:nFeatures].values)
            print(f"Test predictions generated for Fold {Fold + 1}")

        # Restore the order of the indices
        yTrnVal_OOF = yTrnVal_shuffled_pred[restore_indx]

        # Average yTst_pred across various folds (harmonic mean)
        yTst_pred_allFold = (hmean(np.abs(yTst_pred), axis=1) *
                    np.sign(np.mean(yTst_pred, axis=1)))

        # Store predictions
        dict_yTrnVal_OOF[trial][target] = yTrnVal_OOF.copy()
        dict_yTst_pred_allFold[trial][target] = yTst_pred_allFold.copy()

        # Compute CV score
        dict_CV_scores[trial][target] = mape(yTrnVal_shuffled,
                                        yTrnVal_shuffled_pred)

# ============================================
# Average Results Across Trials
# ============================================

print("\n=== AVERAGING ACROSS TRIALS ===")

dict_yTrnVal_avg_final = {}
dict_yTst_avg_final = {}
dict_CV_scores_avg = {}

for target in range(nTargets):
    # Average training OOF predictions across trials
    trial_TrnVal = [dict_yTrnVal_OOF[trial][target] for trial in range(nTrials)]
    dict_yTrnVal_avg_final[target] = (hmean(np.abs(trial_TrnVal), axis=0) *
                              np.sign(np.mean(trial_TrnVal, axis=0)))

    # Average test OOF predictions across trials (use hmean)
    trial_Tst = [dict_yTst_pred_allFold[trial][target] for trial in range(nTrials)]
    dict_yTst_avg_final[target] = (hmean(np.abs(trial_Tst), axis=0) *
                           np.sign(np.mean(trial_Tst, axis=0)))

    # CV scores of averaged predictions
    yTrnVal = (df_XyTrnVal.iloc[:, nFeatures + target].values)
    dict_CV_scores_avg[target] = mape(yTrnVal, dict_yTrnVal_avg_final[target])

    print(f"Target {target + 1}: Avg CV MAPE = {dict_CV_scores_avg[target]:.4f}")


# ============================================
# Save Results
# ============================================

print("\n=== SAVING RESULTS ===")

df_submission = pd.DataFrame()
df_submission['ID'] = range(1, nSamples_Tst + 1)

for target in range(nTargets):
    column_name = f'BlendProperty{target+1}'
    df_submission[column_name] = dict_yTst_avg_final[target]

df_submission.to_csv(ExtractedDATA_DIR + 'RealMLP_submission.csv', index=False)

# ============================================
# Save Trained Models
# ============================================

print("\n=== SAVING TRAINED MODELS ===")

# Save all trained models
with open(Tuning_DIR + 'RealMLP_trained_models.pkl', 'wb') as f:
    pickle.dump(dict_trained_models, f)

print(f"All trained models saved to: {Tuning_DIR}RealMLP_trained_models.pkl")

# Also save individual models for easier access
for trial in range(nTrials):
    for target in range(nTargets):
        for fold in range(nFolds):
            model_filename = f'RealMLP_trial{trial+1}_target{target+1}_fold{fold+1}.pkl'
            model_path = os.path.join(Tuning_DIR, model_filename)
            with open(model_path, 'wb') as f:
                pickle.dump(dict_trained_models[trial][target][fold], f)

print(f"Individual models saved to: {Tuning_DIR}")
print(f"Total models saved: {nTrials * nTargets * nFolds}")

print(f"RealMLP training completed!")
