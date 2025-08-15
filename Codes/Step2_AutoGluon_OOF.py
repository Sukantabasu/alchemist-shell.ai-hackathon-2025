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
File: Step2_AutoGluon_OOF.py
==========================

:Author: Sukanta Basu
:Date: 2025-8-8
:Description: generating out-of-fold (OOF) target values as additional features

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


# ============================================================
# Imports
# ============================================================

import numpy as np
import pandas as pd
import os
import random

from autogluon.tabular import TabularPredictor

# Set random seed for reproducibility
random.seed(7)
np.random.seed(7)


# ============================================================
# User Input
# ============================================================

# AutoGluon quality preset
quality_preset = 'best_quality'

# AutoGluon training time (in seconds)
maxTime = 21600

# Number of input features
nFeatures = 65

# Number of target variables
nTargets = 10


# ============================================================
# Input & Output Directories
# ============================================================

ROOT_DIR = '/data/Sukanta/Works_AIML/2025_SHELL_FuelProperty/'
DATA_DIR = ROOT_DIR + 'DATA/'
ExtractedDATA_DIR = ROOT_DIR + 'ExtractedDATA/'
Tuning_DIR = ROOT_DIR + 'Models/AutoGluon-OOF/'

# Create directory if it doesn't exist
os.makedirs(Tuning_DIR, exist_ok=True)


# ============================================
# Load Processed Training and Testing Data
# ============================================

df_XyTrnVal_org = pd.read_csv(ExtractedDATA_DIR + 'train_processed.csv')
nSamples_TrnVal = df_XyTrnVal_org.shape[0]

df_XTst = pd.read_csv(ExtractedDATA_DIR + 'test_processed.csv')
nSamples_Tst = df_XTst.shape[0]

print(f"Training data shape: {df_XyTrnVal_org.shape}")
print(f"Test data shape: {df_XTst.shape}")

# Extract input features
XTrnVal = df_XyTrnVal_org.iloc[:, 0:nFeatures]

# Initialize predictions array
yTrnVal_OOF = np.zeros((nSamples_TrnVal, nTargets))
yTst = np.zeros((nSamples_Tst, nTargets))


# ============================================
# Iterative Single-target Training using AutoGluon
# ============================================

for target in range(nTargets):
    print(f"\n--- Target {target + 1}/{nTargets} ---")

    # Extract single target from possible nTargets
    yTrnVal = df_XyTrnVal_org.iloc[:, nFeatures + target]

    # Create training dataframe with features and target
    train_data = XTrnVal.copy()
    train_data[f'target_{target}'] = yTrnVal

    # Create unique file path for each target
    target_path = os.path.join(Tuning_DIR, f'target_{target + 1}')
    os.makedirs(target_path, exist_ok=True)

    # Initialize TabularPredictor from AutoGluon
    predictor = TabularPredictor(
        label=f'target_{target}',
        path=target_path,
        eval_metric='mean_absolute_percentage_error',
        problem_type='regression'
    )

    # Train the model
    print("Starting AutoGluon training...")
    predictor.fit(
        train_data,
        time_limit=maxTime,
        presets=quality_preset,
        verbosity=2,
        auto_stack=False,
        dynamic_stacking=False,
        num_bag_folds=8,
        num_bag_sets=5,
        num_stack_levels=2,
        use_bag_holdout=False,
        fit_strategy="sequential",
        ag_args_ensemble={'fold_fitting_strategy': "parallel_local"},
        ds_args={'enable_ray_logging': False}
    )

    print("\n Model Leaderboard:")
    leaderboard = predictor.leaderboard(silent=True)
    print(leaderboard.sort_values("score_val", ascending=False).head())

    # OOF predictions based on training set
    yTrnVal_OOF[:, target] = predictor.predict_oof()

    # Make predictions on test set
    yTst[:, target] = predictor.predict(df_XTst)
    print(f"Test predictions generated for target {target + 1}")

    # Clean up predictor to free memory
    del predictor


# ============================================================
# Save Results
# ============================================================

print("\n=== SAVING RESULTS ===")

# Create dataframes
df_AG_yTrnVal_OOF = pd.DataFrame()
df_AG_yTst = pd.DataFrame()

# Add prediction columns
for i in range(nTargets):
    df_AG_yTrnVal_OOF[f'AG-BlendProperty{i + 1}'] = yTrnVal_OOF[:, i]
    df_AG_yTst[f'AG-BlendProperty{i+1}'] = yTst[:, i]

# Save predictions
AG_OOF_file = os.path.join(ExtractedDATA_DIR, f'AutoGluon_{maxTime}_OOF.csv')
df_AG_yTrnVal_OOF.to_csv(AG_OOF_file, index=False)

AG_Tst_file = os.path.join(ExtractedDATA_DIR, f'AutoGluon_{maxTime}_Tst.csv')
df_AG_yTst.to_csv(AG_Tst_file, index=False)

print(f"AutoGluon training completed!")
