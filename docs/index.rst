Welcome to dpmm's documentation!
====================================

**dpmm** is a Python library that implements state-of-the-art Differentially Private Marginal Models for generating synthetic tabular data.

.. image:: https://github.com/sassoftware/dpmm/workflows/Test%20Suite/badge.svg
   :target: https://github.com/sassoftware/dpmm/actions
   :alt: Tests

.. image:: https://codecov.io/gh/sassoftware/dpmm/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/sassoftware/dpmm
   :alt: Coverage

Overview
--------

Marginal Models have consistently been shown to capture key statistical properties like marginal distributions from the original data and reproduce them in the synthetic data, while Differential Privacy (DP) ensures that individual privacy is rigorously protected.

Key Features
~~~~~~~~~~~~

* **End-to-end DP pipelines** including data preprocessing, generative models, and mechanisms
* **DP data preprocessing** with automatic domain extraction and discretization
* **State-of-the-art DP generative models**: PrivBayes, MST, and AIM
* **Superior utility and performance**
* **Rich functionality** across all models/pipelines
* **DP auditing** of underlying mechanisms and models/pipelines

.. warning::
   **Intended Use**: *dpmm* is designed for research and exploratory use in privacy-preserving synthetic data generation and is not intended for production use in complex, real-world applications.

Quick Start
-----------

.. code-block:: python

   import pandas as pd
   from dpmm.pipelines import MSTPipeline

   # Load your data
   df = pd.read_csv("data.csv")

   # Initialize and fit the model
   model = MSTPipeline(epsilon=1.0, delta=1e-5)
   model.fit(df)

   # Generate synthetic data
   synth_df = model.generate(n_records=1000)


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api
   examples
   contributing
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
