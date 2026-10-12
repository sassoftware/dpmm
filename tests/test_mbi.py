import numpy as np

from dpmm.models.base.mbi import Domain, Factor, FactoredInference


def test_factor_active_single_attribute():
    """Structural zeros on a single attribute should index a 1D factor."""
    domain = Domain(["A"], [4])

    factor = Factor.active(domain, [1, 3])

    assert factor.values[0] == 0
    assert factor.values[2] == 0
    assert factor.values[1] == -np.inf
    assert factor.values[3] == -np.inf


def test_factored_inference_normalizes_single_attribute_structural_zero():
    """Single-attribute structural-zero keys should be stored as cliques."""
    domain = Domain(["col_0", "col_1"], [3, 3])

    inference = FactoredInference(
        domain,
        structural_zeros={"col_0": [1]},
    )

    assert ("col_0",) in inference.structural_zeros
    assert "col_0" not in inference.structural_zeros