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
File: Step1_Preprocess.py
==========================

:Author: Sukanta Basu
:Date: 2025-8-8
:Description: preprocessing and feature engineering

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
import random

# Set random seed for reproducibility
random.seed(7)
np.random.seed(7)


# ============================================================
# Input & Output Directories
# ============================================================

ROOT_DIR = '/data/Sukanta/Works_AIML/2025_SHELL_FuelProperty/'
DATA_DIR = ROOT_DIR + 'DATA/'
ExtractedDATA_DIR = ROOT_DIR + 'ExtractedDATA/'


# ============================================================
# Load Training and Testing Data Provided by the Organizers
# ============================================================

df_XyTrnVal_org = pd.read_csv(DATA_DIR + 'train.csv')
df_XTst_org = pd.read_csv(DATA_DIR + 'test.csv')


# ============================================================
# Feature Engineering
# ============================================================

# Create empty data frames
df_XyTrnVal_mod = pd.DataFrame()
df_XTst_mod = pd.DataFrame()

# Add component fractions
for comp in range(1, 6):
    df_XyTrnVal_mod[f'Component{comp}_fraction'] = (
        df_XyTrnVal_org)[f'Component{comp}_fraction']
    df_XTst_mod[f'Component{comp}_fraction'] = (
        df_XTst_org)[f'Component{comp}_fraction']

# Create volume fraction-weighted input features
for prop in range(1, 11):
    for comp in range(1, 6):
        fraction_col = f'Component{comp}_fraction'
        property_col = f'Component{comp}_Property{prop}'
        contribution_col = f'Component{comp}_Contribution_Property{prop}'
        df_XyTrnVal_mod[contribution_col] = (df_XyTrnVal_org[fraction_col] *
                                             df_XyTrnVal_org[property_col])

        df_XTst_mod[contribution_col] = (df_XTst_org[fraction_col] *
                                             df_XTst_org[property_col])

# Create weighted-averaged input features
for prop in range(1, 11):
    df_XyTrnVal_mod[f'WeightedAvg_Property{prop}'] = (
        sum(df_XyTrnVal_org[f'Component{comp}_fraction'] *
            df_XyTrnVal_org[f'Component{comp}_Property{prop}']
            for comp in range(1, 6)))
    df_XTst_mod[f'WeightedAvg_Property{prop}'] = (
        sum(df_XTst_org[f'Component{comp}_fraction'] *
            df_XTst_org[f'Component{comp}_Property{prop}']
            for comp in range(1, 6)))

# Add targets
for target in range(1, 11):
    df_XyTrnVal_mod[f'BlendProperty{target}'] = df_XyTrnVal_org[f'BlendProperty{target}']


# ============================================================
# Save Processed Data
# ============================================================

df_XyTrnVal_mod.to_csv(ExtractedDATA_DIR + 'train_processed.csv',index=False)
df_XTst_mod.to_csv(ExtractedDATA_DIR + 'test_processed.csv',index=False)
