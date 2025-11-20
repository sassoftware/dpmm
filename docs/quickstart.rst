Quick Start Guide
=================

This guide will help you get started with dpmm in just a few minutes.

Basic Example
-------------

Here's a simple example of how to use dpmm to generate synthetic data:

.. code-block:: python

   import pandas as pd
   from dpmm.pipelines import MSTPipeline

   # Load your data
   df = pd.read_csv("your_data.csv")

   # Initialize the pipeline with privacy parameters
   model = MSTPipeline(
       epsilon=1.0,      # Privacy budget
       delta=1e-5,       # Privacy parameter
       proc_epsilon=0.1  # Preprocessing privacy budget
   )

   # Fit the model to your data
   model.fit(df)

   # Generate synthetic data
   synthetic_df = model.generate(n_records=1000)

   print(synthetic_df.head())


With Domain Information
-----------------------

If you have domain information (bounds for continuous columns), you can provide it:

.. code-block:: python

   import pandas as pd
   import json
   from dpmm.pipelines import MSTPipeline

   # Load data and domain
   df = pd.read_csv("your_data.csv")
   
   with open("domain.json", "r") as f:
       domain = json.load(f)

   # Initialize and fit with domain
   model = MSTPipeline(epsilon=1.0, delta=1e-5)
   model.fit(df, domain)

   # Generate synthetic data
   synthetic_df = model.generate(n_records=1000)


Choosing a Model
----------------

dpmm provides three main models:

**MST (Maximum Spanning Tree)**

Best for: General-purpose synthetic data generation with good performance.

.. code-block:: python

   from dpmm.pipelines import MSTPipeline
   
   model = MSTPipeline(epsilon=1.0, delta=1e-5)


**AIM (Adaptive and Iterative Mechanism)**

Best for: High-quality marginal preservation with adaptive selection.

.. code-block:: python

   from dpmm.pipelines import AIMPipeline
   
   model = AIMPipeline(epsilon=1.0, delta=1e-5)


**PrivBayes (Private Bayesian Network)**

Best for: Preserving dependencies between variables.

.. code-block:: python

   from dpmm.pipelines import PrivBayesPipeline
   
   model = PrivBayesPipeline(epsilon=1.0, delta=1e-5)


Conditional Generation
----------------------

You can generate data conditionally on specific values:

.. code-block:: python

   import pandas as pd
   from dpmm.pipelines import MSTPipeline

   # Fit the model
   model = MSTPipeline(epsilon=1.0, delta=1e-5)
   model.fit(df)

   # Create a partial dataframe with conditions
   conditions = pd.DataFrame({
       'age': [25, 30, 35],
       'gender': ['M', 'F', 'M']
   })

   # Generate remaining columns conditionally
   synthetic_df = model.generate(conditions=conditions)


Deterministic Generation
-------------------------

For reproducible results, use a random seed:

.. code-block:: python

   from dpmm.pipelines import MSTPipeline

   model = MSTPipeline(epsilon=1.0, delta=1e-5)
   model.fit(df)

   # Generate with a fixed seed
   synthetic_df = model.generate(n_records=1000, random_state=42)


Model Serialization
-------------------

Save and load trained models:

.. code-block:: python

   from dpmm.pipelines import MSTPipeline

   # Train and save
   model = MSTPipeline(epsilon=1.0, delta=1e-5)
   model.fit(df)
   model.save("my_model")

   # Load later
   loaded_model = MSTPipeline.load("my_model")
   synthetic_df = loaded_model.generate(n_records=1000)


Next Steps
----------

- Explore the :doc:`user_guide` for more detailed information
- Check out the :doc:`examples` for real-world use cases
- Read the :doc:`api` documentation for all available options
