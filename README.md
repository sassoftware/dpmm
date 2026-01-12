# _dpmm_: Differentially Private Marginal Models, a Library for Synthetic Tabular Data Generation

![Tests](https://github.com/sassoftware/dpmm/workflows/Test%20Suite/badge.svg)
![Coverage](https://raw.githubusercontent.com/sassoftware/dpmm/main/.github/badges/coverage.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2506.00322-b31b1b.svg)](https://arxiv.org/abs/2506.00322)



## Overview

_dpmm_ is a Python library that implements state-of-the-art Differentially Private Marginal Models for generating synthetic tabular data.
Marginal Models have consistently been shown to capture key statistical properties like marginal distributions from the original data and reproduce them in the synthetic data, while Differential Privacy (DP) ensures that individual privacy is rigorously protected.

Summary of main features:
* end-to-end DP pipelines including data preprocessing, generative models, and mechanisms:
   * DP data preprocessing -- 1) data domain is either provided as input or extracted with DP<sup>[paper](https://www.research-collection.ethz.ch/handle/20.500.11850/508570)</sup>, and 2) continous data is discretized with DP (Uniform and PrivTree<sup>[paper](https://arxiv.org/abs/1601.03229)</sup>)
   * state-of-the-art DP generative models relying on the select-measure-generate paradigm<sup>[paper<sub>1</sub>](https://arxiv.org/abs/2108.04978),[paper<sub>2</sub>](https://differentialprivacy.org/synth-data-1/)</sup> and Private-PGM<sup>[paper](https://arxiv.org/abs/1901.09136)</sup> -- PrivBayes<sup>[paper](https://dl.acm.org/doi/10.1145/3134428)</sup>, MST<sup>[paper](https://arxiv.org/abs/2108.04978)</sup>, and AIM<sup>[paper](https://arxiv.org/abs/2201.12677)</sup>
   * floating-point precision of DP mechanisms<sup>[paper](https://arxiv.org/abs/2207.10635)</sup>
* superior utility and performance
* rich functionality across all models/pipelines
* DP auditing of underlying mechanisms and models/pipelines<sup>[paper<sub>1</sub>](https://arxiv.org/abs/2405.10994),[paper<sub>2</sub>](https://dl.acm.org/doi/10.1145/3576915.3616607)</sup>

__NB: Intended Use -- _dpmm_ is designed for research and exploratory use in privacy-preserving synthetic data generation (particularly in simple scenarios such as preserving high-quality 1/2-way marginals in datasets with up to 32 features<sup>[paper<sub>1</sub>](https://arxiv.org/abs/2112.09238),[paper<sub>2</sub>](https://arxiv.org/abs/2305.10994)</sup>) and is not intended for production use in complex, real-world applications.__



## Installation

### Prerequisites

- Python 3.10 or 3.11

### PyPi install

You can also install from PyPi by running: 

```sh
pip install dpmm
```

### Local Install 

To install from the local github repo run the following command: 

```sh
git clone git@github.com:sassoftware/dpmm.git
cd dpmm
poetry install
```

### Tests

To run the unit tests, go to the root of the repository (if installed locally), and use the following command:

```sh
pytest tests/
```



## Functionality

We provide numerous examples demonstrating the features of __dpmm__ across data preprocssing as well as the training and generation of generative models.
The examples are available across all models and model settings, and are accessible from the repository (if installed locally).


__NB: the general intent of this package is to be used through the pipeline layer to guarantee that no privacy leakage is occuring, it is possible to use the models directly but in that instance providing a domain is a requirement to ensure privacy guarantees.__

### Preprocessing
The provided generative pipelines combine automatic DP descritization preprocessing with a generative model and allows for the following features:

| Feature | Description | Example |
| --- | --- | --- |
| __dtype support__ | the following pandas data types are supported natively: `datetime`, `timedelta`, `float`, `int`, `category`, `bool`. | [Dtypes example](https://github.com/sassoftware/dpmm/tree/main/examples/example_dtypes.ipynb) |
|__null-value support__ | missing values are supported and will be reproduced accordingly if present in any column within the real data. | |
|__automatic discretisation__ | while the default discretisation strategy used by _dpmm_ is `priv-tree` a more typical `uniform` strategy is also availble, they can both be combined with an `'auto'` mode which will attempt to identify the optimal number of bins for each numerical column column. | |


### Model Features

| Feature | Description | Example |
| --- | --- | --- |
| __domain compression__ | a `compress` flag can be set to `True` to ensure the discretised domain is compressed to improve the privacy budget / data quality trade-off. |  |
|__model size control__ | a `max_model_size` parameter that ensures the memory footprint of the selected marginals remains lower than the specified upper threshold. | [Max Memory example](https://github.com/sassoftware/dpmm/tree/main/examples/example_memory.ipynb) |
|__model serialisation__ | pipelines can be serialised to / deserialised from disk by provided a valid folder to store the model to. | [Serialisation example](https://github.com/sassoftware/dpmm/tree/main/examples/example_serialisation.ipynb) |


### Generation Features

| Feature | Description | Example |
| --- | --- | --- |
| __conditional generation__ | at generation time, it is also possible to provide a partial dataframe containing only some of the columns, in that case the generative pipeline will conditionally generate the remaining columns. | [Conditional Generation example](https://github.com/sassoftware/dpmm/tree/main/examples/example_conditional.ipynb) |
| __deterministic generation__ | when a `random_state` value is provided at generation time, the generative process becomes deterministic assuming the same input parameters are provided. | [Random State example](https://github.com/sassoftware/dpmm/tree/main/examples/example_seed.ipynb) |

### Models
The implemented models include:

| Method | Description | Reference | Example | 
|--- | --- | --- | --- | 
|**PrivBayes+PGM**|  Differentialy Private Bayesian Network. | [PrivBayes: Private Data Release via Bayesian Networks](https://dl.acm.org/doi/10.1145/3134428)| [PrivBayes example](https://github.com/sassoftware/dpmm/tree/main/examples/example_privbayes.ipynb) |
|**MST**|  Maximum Spanning Tree. | [Winning the NIST Contest: A scalable and general approach to differentially private synthetic data](https://arxiv.org/abs/2108.04978)| [MST example](https://github.com/sassoftware/dpmm/tree/main/examples/example_mst.ipynb) | 
|**AIM**|  Adaptive and Iterative Mechanism. | [AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data](https://arxiv.org/abs/2201.12677)| [AIM example](https://github.com/sassoftware/dpmm/tree/main/examples/example_aim.ipynb) |

__NB: All models rely on the select-measure-generate paradigm<sup>[paper<sub>1</sub>](https://arxiv.org/abs/2108.04978),[paper<sub>2</sub>](https://differentialprivacy.org/synth-data-1/)</sup> and Private-PGM<sup>[paper](https://arxiv.org/abs/1901.09136)</sup>.__



## Getting Started

To get started with using the _dpmm_, follow the steps below:

1. Import the necessary modules and load your data:
   ```python
   import pandas as pd
   import json
   from dpmm.pipelines import MSTPipeline


   wine_dir = Path().parent / "wine"

   df = pd.read_pickle(wine_dir / "wine.pkl.gz")
   with (wine_dir / "wine_bounds.json").open("r") as f:
      domain = json.load(f)
   ```

2. Initialize and fit a model:

   ```python
   model = MSTPipeline(
      # Generator Parameters
      epsilon=1.0, 
      delta=1e-5,
      # Discretiser Parametrs
      proc_epsilon=0.1,
   )
   model.fit(df, domain)
   ```

3. Generate synthetic data:
   ```python
   synth_df = model.generate(n_records=100)
   print(synth_df)
   """
         type  fixed acidity  volatile acidity  citric acid  residual sugar   chlorides free sulfur dioxide  total sulfur dioxide   density        pH   sulphates    alcohol quality  
      0  white       5.288142          0.190330     0.212473        1.402665    0.032305            37.097305             60.585301  0.990234  2.998241    0.658841  12.467682       1  
      1  white       5.956364          0.225099     0.210124       15.968057    0.043620            70.073909            202.689578  0.995807  3.198247    0.318414  10.290390       0  
      2  white       5.315535          0.341091     0.247268        0.628240    0.024938            52.468176            104.892353  0.990975  3.161218    0.971699  11.181373       1  
      3  white       7.879125          0.234170     0.275704        3.711610    0.039565            68.977194            163.380550  1.005989  3.068622    0.798520   8.075999       0  
      4  white       6.981342          0.358461     0.337705        3.600390    0.050450            51.567452            134.896467  0.996149  3.272745    0.599021  10.200400       0  

   """
   ```



### Troubleshooting

If you encounter any issues, please check the following:

- Ensure that all required packages are installed.
- Verify that your data does not contain missing values or non-integer columns if using certain models.
- Check the model parameters and ensure they are set correctly.



## Contributing

Maintainers are accepting patches and contributions to this project.
Please read [CONTRIBUTING.md](https://github.com/sassoftware/dpmm/tree/main/CONTRIBUTING.md) for details about submitting contributions to this project.



## License

This project is licensed under the [Apache 2.0 License](https://github.com/sassoftware/dpmm/tree/main/LICENSE).
This project also uses code snippets from the following projects: 
- [private-pgm](https://github.com/ryan112358/private-pgm): Apache 2.0
- [opendp](https://github.com/opendp/smartnoise-sdk): MIT License
- [ektelo](https://github.com/ektelo/ektelo): Apache 2.0



## Additional Resources

* [SAS Global Forum Papers](https://www.sas.com/en_us/events/sas-global-forum.html)
* [SAS Communities](https://communities.sas.com/)



## Citing

If you use this code, please cite the associated paper:
```
@inproceedings{mahiou2025dpmm,
  title={{dpmm: Differentially Private Marginal Models, a Library for Synthetic Tabular Data Generation}},
  author={Mahiou, Sofiane and Dizche, Amir and Nazari, Reza and Wu, Xinmin and Abbey, Ralph and Silva, Jorge and Ganev, Georgi},
  booktitle={TPDP},
  year={2025}
}
```
