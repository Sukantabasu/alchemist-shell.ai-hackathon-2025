Documentation of ALCHEMIST
==========================

ALCHEMIST: (A)IML-powered (L)ow-(c)ost, (H)igh-octane (E)co-fuel (M)ixture (I)dentification (S)trategy & (T)oolkit

Problem Statement
-----------------

The Shell AI Hackathon 2025 focuses on predicting fuel blend properties using machine learning techniques. This challenge involves predicting multiple target properties (BlendProperty1 through BlendProperty10) of fuel blends based on the composition and properties of individual components.

Dataset Overview
-----------------

The problem involves:
- **Input Features**: 5 fuel components, each with:
  - Volume fraction in the blend
  - 10 individual properties per component (Component1\_Property1 through Component5\_Property10)
- **Target Variables**: 10 blend properties that need to be predicted
- **Training Data**: Historical fuel blend data with known outcomes, sample size = 2000
- **Test Data**: New fuel blend compositions requiring property predictions, sample size = 500

Challenging Issues
------------------

1. Multi-Target Regression Complexity

The problem presents a multi-target regression challenge where 10 different blend properties must be predicted simultaneously. The blend properties are interdependent, requiring models that can capture these relationships. 

2. Training Data Limitation

Due to the limited samples (only 2000) in the training set, traditional hyper-parameter tuning-based multioutput regression approaches are impractical.  


Propopsed Framework
-------------------


.. toctree::
   :maxdepth: 2
   :caption: ALCHEMIST's Recipe:

   README
   notebooks/index

AI Assistance 
--------------

Claude.AI (Anthropic) is used for documentation, code restructuring, and performance optimization


Installation
------------

To install the required dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Usage
-----

The project consists of several steps:

1. **Step 1**: Preprocessing and feature engineering
2. **Step 2**: AutoGluon out-of-fold predictions
3. **Step 3**: RealMLP model training
4. **Step 4**: TabPFN model training
5. **Step 5**: Model combination

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
