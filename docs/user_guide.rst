User Guide
==========

This comprehensive guide covers all aspects of using dpmm.

.. contents::
   :local:
   :depth: 2


Understanding Differential Privacy
-----------------------------------

Differential Privacy (DP) is a rigorous mathematical framework that provides formal privacy guarantees. In the context of synthetic data generation, DP ensures that individual records in your dataset cannot be identified or reconstructed from the synthetic data.

Privacy Parameters
~~~~~~~~~~~~~~~~~~

**Epsilon (ε)**

- The privacy budget parameter
- Lower values = stronger privacy protection but potentially lower data quality
- Typical values: 0.1 to 10
- Recommended starting point: 1.0

**Delta (δ)**

- The failure probability
- Should be much smaller than 1/n where n is the dataset size
- Typical values: 1e-5 to 1e-9
- For a dataset with 10,000 records, use δ ≤ 1e-4


Data Preprocessing
------------------

dpmm automatically preprocesses your data with differential privacy guarantees.

Supported Data Types
~~~~~~~~~~~~~~~~~~~~

dpmm natively supports the following pandas data types:

- ``int``: Integer columns
- ``float``: Continuous numerical columns
- ``category``: Categorical columns
- ``bool``: Boolean columns
- ``datetime``: Date and time columns
- ``timedelta``: Time duration columns

Missing Values
~~~~~~~~~~~~~~

Missing values (NaN) are automatically handled:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from dpmm.pipelines import MSTPipeline

   # Data with missing values
   df = pd.DataFrame({
       'age': [25, 30, np.nan, 35, 40],
       'income': [50000, np.nan, 60000, 70000, 80000],
       'city': ['NYC', 'LA', 'NYC', np.nan, 'LA']
   })

   model = MSTPipeline(epsilon=1.0, delta=1e-5)
   model.fit(df)
   
   # Synthetic data will preserve missing value patterns
   synthetic_df = model.generate(n_records=100)


Discretization
~~~~~~~~~~~~~~

Continuous columns are automatically discretized using DP algorithms:

**PrivTree (default)**

Adaptive discretization that learns optimal bin boundaries:

.. code-block:: python

   model = MSTPipeline(
       epsilon=1.0,
       delta=1e-5,
       discretization='priv-tree',
       proc_epsilon=0.1  # Privacy budget for preprocessing
   )


**Uniform**

Equal-width discretization:

.. code-block:: python

   model = MSTPipeline(
       epsilon=1.0,
       delta=1e-5,
       discretization='uniform',
       n_bins=10  # Number of bins per column
   )


**Auto Mode**

Automatically determines optimal number of bins:

.. code-block:: python

   model = MSTPipeline(
       epsilon=1.0,
       delta=1e-5,
       discretization='uniform',
       n_bins='auto'
   )


Domain Information
~~~~~~~~~~~~~~~~~~

For better results, provide domain information (bounds for continuous columns):

.. code-block:: python

   domain = {
       'age': (0, 120),
       'income': (0, 1000000),
       'num_children': (0, 10)
   }

   model = MSTPipeline(epsilon=1.0, delta=1e-5)
   model.fit(df, domain=domain)


Model Selection
---------------

MST (Maximum Spanning Tree)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best for:** General-purpose synthetic data generation

**Advantages:**
- Good balance between quality and performance
- Scalable to larger datasets
- Preserves pairwise relationships well

**Example:**

.. code-block:: python

   from dpmm.pipelines import MSTPipeline

   model = MSTPipeline(
       epsilon=1.0,
       delta=1e-5,
       proc_epsilon=0.1,
       degree=2,  # Maximum degree in the spanning tree
       max_model_size=1000  # Memory constraint (MB)
   )


AIM (Adaptive and Iterative Mechanism)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best for:** High-quality marginal preservation

**Advantages:**
- Adaptive selection of marginals
- Excellent marginal accuracy
- Iterative refinement

**Example:**

.. code-block:: python

   from dpmm.pipelines import AIMPipeline

   model = AIMPipeline(
       epsilon=1.0,
       delta=1e-5,
       proc_epsilon=0.1,
       degree=2,
       num_marginals=None,  # Automatic selection
       max_cells=10000  # Maximum cells per marginal
   )


PrivBayes (Private Bayesian Network)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best for:** Preserving complex dependencies

**Advantages:**
- Models conditional dependencies
- Good for datasets with strong feature interactions
- Bayesian network structure learning

**Example:**

.. code-block:: python

   from dpmm.pipelines import PrivBayesPipeline

   model = PrivBayesPipeline(
       epsilon=1.0,
       delta=1e-5,
       proc_epsilon=0.1,
       degree=2,
       max_parents=2  # Maximum parents per node in Bayesian network
   )


Advanced Features
-----------------

Memory Control
~~~~~~~~~~~~~~

Limit memory usage of the learned model:

.. code-block:: python

   model = MSTPipeline(
       epsilon=1.0,
       delta=1e-5,
       max_model_size=500  # Maximum 500 MB
   )


Domain Compression
~~~~~~~~~~~~~~~~~~

Compress the discretized domain to improve quality:

.. code-block:: python

   model = MSTPipeline(
       epsilon=1.0,
       delta=1e-5,
       compress=True  # Enable compression
   )


Model Serialization
~~~~~~~~~~~~~~~~~~~

Save and load trained models:

.. code-block:: python

   # Save model
   model.fit(df)
   model.save("models/my_model")

   # Load model
   from dpmm.pipelines import MSTPipeline
   loaded_model = MSTPipeline.load("models/my_model")
   synthetic_df = loaded_model.generate(n_records=1000)


Conditional Generation
~~~~~~~~~~~~~~~~~~~~~~

Generate data conditionally on specific column values:

.. code-block:: python

   import pandas as pd

   # Train model
   model.fit(df)

   # Create conditions
   conditions = pd.DataFrame({
       'age': [25, 30, 35],
       'gender': ['M', 'F', 'M']
   })

   # Generate remaining columns
   synthetic_df = model.generate(conditions=conditions)


Reproducibility
~~~~~~~~~~~~~~~

Use random seeds for deterministic generation:

.. code-block:: python

   # Training is deterministic by default
   model.fit(df)

   # Generation with fixed seed
   synthetic_df1 = model.generate(n_records=1000, random_state=42)
   synthetic_df2 = model.generate(n_records=1000, random_state=42)
   
   # synthetic_df1 and synthetic_df2 will be identical


Best Practices
--------------

1. **Start with reasonable privacy budgets**
   
   - Begin with ε=1.0 and adjust based on your privacy requirements
   - Allocate 10-20% of budget to preprocessing (proc_epsilon)

2. **Provide domain information when possible**
   
   - Helps with better discretization
   - Improves synthetic data quality

3. **Choose the right model for your use case**
   
   - MST: General-purpose, good default choice
   - AIM: When marginal accuracy is critical
   - PrivBayes: When preserving dependencies is important

4. **Monitor memory usage**
   
   - Use max_model_size for large datasets
   - Consider degree parameter to limit marginal sizes

5. **Validate synthetic data quality**
   
   - Compare marginal distributions
   - Check correlations between variables
   - Evaluate downstream task performance


Troubleshooting
---------------

**Issue:** Out of memory errors

**Solution:** 
- Reduce max_model_size
- Decrease degree parameter
- Use fewer bins in discretization

**Issue:** Poor quality synthetic data

**Solution:**
- Increase privacy budget (ε)
- Provide domain information
- Increase number of bins
- Try a different model

**Issue:** Slow training time

**Solution:**
- Use MST instead of AIM or PrivBayes
- Reduce number of bins
- Limit max_model_size
- Use domain compression
