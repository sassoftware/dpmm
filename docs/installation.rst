Installation
============

Prerequisites
-------------

- Python 3.10 or 3.11

From PyPI
---------

The easiest way to install dpmm is via pip:

.. code-block:: bash

   pip install dpmm

From Source
-----------

To install from the GitHub repository:

.. code-block:: bash

   git clone git@github.com:sassoftware/dpmm.git
   cd dpmm
   poetry install

Development Installation
-------------------------

To install with development dependencies:

.. code-block:: bash

   git clone git@github.com:sassoftware/dpmm.git
   cd dpmm
   poetry install --with dev

Verifying Installation
----------------------

To verify that the installation was successful, run the tests:

.. code-block:: bash

   pytest tests/

Or import the package in Python:

.. code-block:: python

   import dpmm
   from dpmm.pipelines import MSTPipeline, AIMPipeline, PrivBayesPipeline
   
   print(dpmm.__version__)

Dependencies
------------

The main dependencies are:

- pandas >= 2.1.0
- numpy >= 1.26.4
- scipy >= 1.15.2
- scikit-learn >= 1.5.0
- tqdm
- networkx < 3.0
- disjoint-set
- opendp >= 0.12.1

All dependencies will be automatically installed when you install dpmm.
