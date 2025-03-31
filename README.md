# Differentially Private Marginal Models (dpmm)

## Overview

This project provides implementations of differentially private generative models (dpmm) for creating synthetic data. The models ensure that the generated data maintains privacy guarantees while preserving the statistical properties of the original data. The implemented models include:

- AIMGM (Adaptive Information Maximization Generative Model)
- PrivBayesGM (Private Bayesian Network Generative Model)
- MSTGM (Minimum Spanning Tree Generative Model)

### Prerequisites

- Python 3.11 or higher
-

## Installation

```
pip install dpmm
```

### Getting Started

To get started with using the differentially private generative models, follow the steps below:

1. Import the necessary modules and load your data:
   ```python
   import pandas as pd
   from dpmm.pipelines import MSTEngine

   # Load your data
   data = pd.read_csv('your_data.csv')
   ```

2. Initialize and fit a model:

   ```python
   model = MSTEngine(
      # Generator Parameters
      synth_epsilon=1.0, 
      gen_kwargs=dict(domain=None, compress=True, max_model_size=80), 
      # Discretiser Parametrs
      proc_epsilon=0.1,
   )
   model.fit(data)
   ```

3. Generate synthetic data:
   ```python
   synthetic_data = model.generate(n_records=100)
   print(synthetic_data)
   ```

### Running

To run the unit tests for the models, use the following command:
```sh
pytest tests/
```

### Examples

Here is an example of using the PrivBayesGM model:

```python
import pandas as pd
from dpmm.models import PrivBayesGM

# Load your data
data = pd.read_csv('your_data.csv')

# Initialize and fit the model
model = PrivBayesGM(epsilon=1.0, domain=None, compress=True, max_model_size=80)
model.fit(data)

# Generate synthetic data
synthetic_data = model.generate(n_records=100)
print(synthetic_data)
```

### Troubleshooting

If you encounter any issues, please check the following:

- Ensure that all required packages are installed.
- Verify that your data does not contain missing values or non-integer columns if using certain models.
- Check the model parameters and ensure they are set correctly.

## Contributing

Maintainers are accepting patches and contributions to this project.
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details about submitting contributions to this project.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
This project also uses code snippets from the following projects : 
- [private-pgm](https://github.com/ryan112358/private-pgm): Apache 2.0
- [ektelo](https://github.com/ektelo/ektelo): Apache 2.0

## Additional Resources

* [Differential Privacy Overview](https://en.wikipedia.org/wiki/Differential_privacy)
* __Models__: [MST](https://arxiv.org/pdf/2108.04978), [AIM](https://arxiv.org/pdf/2201.12677), [PrivBayes](http://dimacs.rutgers.edu/~graham/pubs/papers/PrivBayes.pdf)
* __Processing__: [PrivTree](https://arxiv.org/pdf/1601.03229)
* [SAS Global Forum Papers](https://www.sas.com/en_us/events/sas-global-forum.html)
* [SAS Communities](https://communities.sas.com/)