Examples
========

This page provides detailed examples of using dpmm in various scenarios.

Basic Example
-------------

Wine Dataset
~~~~~~~~~~~~

This example demonstrates basic usage with the wine quality dataset:

.. code-block:: python

   import pandas as pd
   import json
   from pathlib import Path
   from dpmm.pipelines import MSTPipeline

   # Load data
   wine_dir = Path("examples/wine")
   df = pd.read_pickle(wine_dir / "wine.pkl.gz")
   
   with (wine_dir / "wine_bounds.json").open("r") as f:
       domain = json.load(f)

   # Initialize and fit model
   model = MSTPipeline(
       epsilon=1.0, 
       delta=1e-5,
       proc_epsilon=0.1,
   )
   model.fit(df, domain)

   # Generate synthetic data
   synth_df = model.generate(n_records=1000)
   print(synth_df.head())


Data Types Example
------------------

Handling Different Data Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dpmm supports various pandas data types:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from datetime import datetime, timedelta
   from dpmm.pipelines import MSTPipeline

   # Create dataset with different types
   df = pd.DataFrame({
       'int_col': [1, 2, 3, 4, 5],
       'float_col': [1.5, 2.5, 3.5, 4.5, 5.5],
       'category_col': pd.Categorical(['A', 'B', 'A', 'C', 'B']),
       'bool_col': [True, False, True, False, True],
       'datetime_col': pd.date_range('2023-01-01', periods=5),
       'timedelta_col': [timedelta(days=i) for i in range(5)]
   })

   # Train model
   model = MSTPipeline(epsilon=1.0, delta=1e-5)
   model.fit(df)

   # Generate synthetic data - types are preserved
   synth_df = model.generate(n_records=100)
   print(synth_df.dtypes)


Conditional Generation
----------------------

Generating Specific Subpopulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate synthetic data for specific conditions:

.. code-block:: python

   import pandas as pd
   from dpmm.pipelines import MSTPipeline

   # Load and fit model
   df = pd.read_csv("customer_data.csv")
   model = MSTPipeline(epsilon=1.0, delta=1e-5)
   model.fit(df)

   # Generate data for specific age groups
   conditions = pd.DataFrame({
       'age_group': ['18-25', '26-35', '36-45', '46-55', '56+'] * 200
   })

   # Generate remaining features conditionally
   synth_df = model.generate(conditions=conditions)
   print(synth_df.groupby('age_group').size())


Memory Management
-----------------

Large Dataset Example
~~~~~~~~~~~~~~~~~~~~~

Handle large datasets with memory constraints:

.. code-block:: python

   import pandas as pd
   from dpmm.pipelines import MSTPipeline

   # Load large dataset
   df = pd.read_csv("large_dataset.csv")
   print(f"Dataset size: {len(df)} rows, {len(df.columns)} columns")

   # Configure memory limits
   model = MSTPipeline(
       epsilon=1.0,
       delta=1e-5,
       max_model_size=1000,  # Limit to 1GB
       compress=True,  # Enable domain compression
       degree=2  # Limit marginal degree
   )

   model.fit(df)
   synth_df = model.generate(n_records=len(df))


Model Serialization
-------------------

Save and Load Models
~~~~~~~~~~~~~~~~~~~~

Train once, generate multiple times:

.. code-block:: python

   from dpmm.pipelines import MSTPipeline
   import pandas as pd

   # Train and save
   df = pd.read_csv("training_data.csv")
   model = MSTPipeline(epsilon=1.0, delta=1e-5)
   model.fit(df)
   model.save("saved_models/my_mst_model")

   # Later: Load and generate
   loaded_model = MSTPipeline.load("saved_models/my_mst_model")
   
   # Generate multiple batches
   batch1 = loaded_model.generate(n_records=1000, random_state=42)
   batch2 = loaded_model.generate(n_records=1000, random_state=43)
   batch3 = loaded_model.generate(n_records=1000, random_state=44)


Reproducibility
---------------

Deterministic Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure reproducible results:

.. code-block:: python

   from dpmm.pipelines import MSTPipeline
   import pandas as pd

   df = pd.read_csv("data.csv")

   # Training
   model = MSTPipeline(epsilon=1.0, delta=1e-5)
   model.fit(df)  # Training is deterministic

   # Reproducible generation
   synth1 = model.generate(n_records=1000, random_state=42)
   synth2 = model.generate(n_records=1000, random_state=42)
   
   # These will be identical
   assert synth1.equals(synth2)


Model Comparison
----------------

Comparing Different Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare MST, AIM, and PrivBayes:

.. code-block:: python

   import pandas as pd
   from dpmm.pipelines import MSTPipeline, AIMPipeline, PrivBayesPipeline

   # Load data
   df = pd.read_csv("data.csv")

   # Same privacy parameters for fair comparison
   epsilon = 1.0
   delta = 1e-5
   proc_epsilon = 0.1

   # Train all three models
   mst_model = MSTPipeline(epsilon=epsilon, delta=delta, proc_epsilon=proc_epsilon)
   mst_model.fit(df)

   aim_model = AIMPipeline(epsilon=epsilon, delta=delta, proc_epsilon=proc_epsilon)
   aim_model.fit(df)

   pb_model = PrivBayesPipeline(epsilon=epsilon, delta=delta, proc_epsilon=proc_epsilon)
   pb_model.fit(df)

   # Generate synthetic data
   mst_synth = mst_model.generate(n_records=1000)
   aim_synth = aim_model.generate(n_records=1000)
   pb_synth = pb_model.generate(n_records=1000)

   # Compare quality (example metrics)
   from sklearn.metrics import mean_absolute_error
   
   def compare_marginals(real, synth, column):
       real_dist = real[column].value_counts(normalize=True).sort_index()
       synth_dist = synth[column].value_counts(normalize=True).sort_index()
       return mean_absolute_error(real_dist, synth_dist)

   for col in df.select_dtypes(include=['category', 'object']).columns:
       mst_err = compare_marginals(df, mst_synth, col)
       aim_err = compare_marginals(df, aim_synth, col)
       pb_err = compare_marginals(df, pb_synth, col)
       
       print(f"{col}:")
       print(f"  MST: {mst_err:.4f}")
       print(f"  AIM: {aim_err:.4f}")
       print(f"  PrivBayes: {pb_err:.4f}")


Privacy Budget Allocation
--------------------------

Optimizing Privacy Budget Split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Experiment with different budget allocations:

.. code-block:: python

   import pandas as pd
   from dpmm.pipelines import MSTPipeline

   df = pd.read_csv("data.csv")
   total_epsilon = 1.0

   # Try different preprocessing budgets
   for proc_ratio in [0.05, 0.1, 0.15, 0.2]:
       proc_eps = total_epsilon * proc_ratio
       gen_eps = total_epsilon - proc_eps
       
       model = MSTPipeline(
           epsilon=gen_eps,
           delta=1e-5,
           proc_epsilon=proc_eps
       )
       model.fit(df)
       synth = model.generate(n_records=1000)
       
       # Evaluate quality
       print(f"Preprocessing: {proc_eps:.2f}, Generation: {gen_eps:.2f}")
       # Add your quality metrics here


Real-World Use Cases
--------------------

Healthcare Data
~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from dpmm.pipelines import PrivBayesPipeline

   # Load sensitive healthcare data
   df = pd.read_csv("patient_records.csv")

   # Define domain for medical measurements
   domain = {
       'age': (0, 120),
       'blood_pressure_systolic': (60, 200),
       'blood_pressure_diastolic': (40, 130),
       'heart_rate': (40, 200),
       'bmi': (10, 60)
   }

   # Use strong privacy guarantees
   model = PrivBayesPipeline(
       epsilon=0.5,  # Strong privacy
       delta=1e-6,   # Very low failure probability
       proc_epsilon=0.05
   )

   model.fit(df, domain=domain)
   
   # Generate synthetic patient records
   synthetic_patients = model.generate(n_records=10000)


Financial Data
~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from dpmm.pipelines import MSTPipeline

   # Load transaction data
   df = pd.read_csv("transactions.csv")

   # Domain for financial features
   domain = {
       'transaction_amount': (0, 100000),
       'account_balance': (0, 1000000),
       'credit_score': (300, 850)
   }

   model = MSTPipeline(
       epsilon=1.0,
       delta=1e-5,
       proc_epsilon=0.1,
       compress=True
   )

   model.fit(df, domain=domain)
   synthetic_transactions = model.generate(n_records=50000)


More Examples
-------------

For more examples, including Jupyter notebooks, visit:

- `GitHub Examples Directory <https://github.com/sassoftware/dpmm/tree/main/examples>`_
- Example notebooks with the wine dataset
- HMEQ dataset example
- Advanced conditional generation examples
