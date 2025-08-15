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
File: Step5_Ensemble.py
==========================

:Author: Sukanta Basu
:Date: 2025-8-15
:Description: Ensemble predictions from RealMLP and TabPFN models

Overall strategy:
Step 1: preprocessing and feature generation
Step 2: Use AutoGluon to generate OOF predictions for each target separately.
These predictions will be used as additional input features in steps 3 and 4.
Step 3: Train the RealMLP model with processed input (step 1) + ten
AutoGluon-OOFs (step 2). These additional features will capture the correlation
among targets effectively.
Step 4: Similar to step 3 except use the TabPFN model.
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

# Set random seed for reproducibility
random.seed(7)
np.random.seed(7)

# ============================================
# Input & Output Directories
# ============================================

ROOT_DIR = '/data/Sukanta/Works_AIML/2025_SHELL_FuelProperty/'
DATA_DIR = ROOT_DIR + 'DATA/'
ExtractedDATA_DIR = ROOT_DIR + 'ExtractedDATA/'

# ============================================
# Load Predictions from RealMLP and TabPFN
# ============================================

print("=== LOADING PREDICTIONS ===")

# Load RealMLP predictions
df_realmlp = pd.read_csv(ExtractedDATA_DIR + 'RealMLP_submission.csv')
print(f"RealMLP predictions shape: {df_realmlp.shape}")
print(f"RealMLP columns: {list(df_realmlp.columns)}")

# Load TabPFN predictions
df_tabpfn = pd.read_csv(ExtractedDATA_DIR + 'TabPFN_submission.csv')
print(f"TabPFN predictions shape: {df_tabpfn.shape}")
print(f"TabPFN columns: {list(df_tabpfn.columns)}")


# ============================================
# Create Ensemble Predictions
# ============================================

print("\n=== CREATING ENSEMBLE PREDICTIONS ===")

# Initialize ensemble dataframe
df_ensemble = pd.DataFrame()
df_ensemble['ID'] = df_realmlp['ID'].copy()

# Use TabPFN for targets 1-4, RealMLP for targets 5-10
for target in range(1, 11):
    column_name = f'BlendProperty{target}'

    if target <= 4:
        # Use TabPFN for targets 1-4
        df_ensemble[column_name] = df_tabpfn[column_name].copy()
        print(f"Target {target}: Using TabPFN predictions")
    else:
        # Use RealMLP for targets 5-10
        df_ensemble[column_name] = df_realmlp[column_name].copy()
        print(f"Target {target}: Using RealMLP predictions")


# ============================================
# Save Ensemble Predictions
# ============================================

print("\n=== SAVING ENSEMBLE PREDICTIONS ===")

ensemble_file = ExtractedDATA_DIR + 'Ensemble_submission.csv'
df_ensemble.to_csv(ensemble_file, index=False)
