ALCHEMIST Documentation
=======================

ALCHEMIST: (A)IML-powered (L)ow-(c)ost, (H)igh-octane (E)co-fuel (M)ixture (I)dentification (S)trategy & (T)oolkit

Problem Statement
-----------------

The Shell AI Hackathon 2025 focuses on predicting fuel blend properties using machine learning techniques. This challenge involves predicting multiple target properties (BlendProperty1 through BlendProperty10) of fuel blends based on the composition and properties of individual components.


Challenges
----------


Propopsed Framework
-------------------


.. toctree::
   :maxdepth: 2
   :caption: ALCHEMIST's Recipe:

   README
   notebooks/index

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
