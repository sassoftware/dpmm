import numpy as np
from opendp.measurements import make_laplace, make_gaussian
from opendp.mod import enable_features
import opendp.prelude as dp


def gaussian_noise(sigma, size=None):
    enable_features("floating-point", "contrib")
    input_domain = dp.atom_domain(T=float)
    input_metric = dp.absolute_distance(T=float)
    meas = make_gaussian(input_domain, input_metric, sigma)
    if size is None:
        return meas(0.0)
    elif isinstance(size, np.number):
        return np.array([meas(0.0) for _ in range(size)])
    else:
        _size = np.prod(size)
        measurements = [meas(0.0) for _ in range(_size)]
        return np.array(measurements).reshape(size)


def laplace_noise(scale, size=None):
    enable_features("floating-point", "contrib")
    input_domain = dp.atom_domain(T=float)
    input_metric = dp.absolute_distance(T=float)
    meas = make_laplace(input_domain, input_metric, scale)
    if size is None:
        return meas(0.0)
    else:
        return [meas(0.0) for _ in range(size)]
