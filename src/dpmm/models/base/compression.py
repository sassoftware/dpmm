import pandas as pd
import numpy as np
from scipy import sparse
from dpmm.models.base.mbi import Dataset, Domain


class DomainCompressor:
    def __init__(self, prng=None, structural_zeros=None, zeros_only=False):
        self.supports = {}
        self.zeros = {}
        if prng is None:
            prng = np.random.RandomState()
        self.prng = prng
        # structural zeros
        if structural_zeros is None:
            structural_zeros = {}
        self.structural_zeros = structural_zeros
        self.zeros_only = zeros_only

    def fit(self, measurements, flatten=False):
        self.supports = {}
        return [self.fit_measure(measure, flatten=flatten) for measure in measurements]

    def fit_measure(self, measure, flatten=False):
        Q, y, sigma, proj = measure

        I2 = np.ones(y.size)

        for axis, col in enumerate(proj):
            y_col = np.sum(y, axis=tuple([ax for ax in range(len(proj)) if ax != axis]))

            if col not in self.supports:
                # Structural Zeros
                self.zeros[col] = np.zeros(y_col.shape[0], dtype=bool)
                if col in self.structural_zeros:
                    struct = np.array(self.structural_zeros[col]).astype(int)
                    self.zeros[col][struct] = True

                self.supports[col] = ~self.zeros[col]

                if not self.zeros_only:
                    if pd.isnull(sigma):
                        sup = np.ones(y_col.shape[0], dtype=bool)
                    else:
                        sup = y_col >= (3 * sigma)

                        if sup.sum() == 0:
                            # TODO: Add warning compression for target column failed
                            sup = np.ones(y_col.shape[0], dtype=bool)

                    self.supports[col] &= sup

            sup = self.supports[col]

            if self.supports[col].sum() == y.shape[axis]:
                continue
            # need to re-express measurement over the new domain
            else:
                # Select Counts to Preserve
                preserved_idx = np.arange(y_col.shape[0])[sup]
                y_new = np.take(y, indices=preserved_idx, axis=axis)

                # Concatenating Count
                if not self.zeros_only:
                    normaliser = np.sqrt(y_col.shape[0] - sup.sum() + 1)
                    agg_idx = np.arange(y_col.shape[0])[~sup]
                    y_agg = (
                        np.expand_dims(
                            np.take(y, indices=agg_idx, axis=axis).sum(axis=axis),
                            axis=axis,
                        )
                        / normaliser
                    )
                    y = np.concatenate((y_new, y_agg), axis=axis)
                else:
                    y = y_new

                # Correct Weight
                I2 = np.ones(y.size)
                if len(proj) == 1 and not self.zeros_only:
                    I2[-1] = 1.0 / normaliser

        Q = sparse.diags(I2)
        
        if flatten:
            y = y.flatten()

        return (Q, y, sigma, proj)

    def transform_col(self, df, col):
        as_df = isinstance(df, pd.DataFrame)
        if not as_df:
            domain = df.domain.config
            df = df.df

        support = self.supports[col]
        size = support.sum()
        newdom = int(size)

        mapped = df[col].copy()

        if col in self.zeros:
            # TODO : Add warning zeros only
            is_zero = self.zeros[col][mapped]
            if is_zero.any():
                mapped.loc[is_zero] = (
                    mapped.loc[~is_zero]
                    .sample(n=is_zero.sum(), replace=True)
                    .to_numpy()
                )

        mapping = {}
        idx = 0
        for i in range(support.size):
            mapping[i] = size
            if support[i]:
                mapping[i] = idx
                idx += 1
        assert idx == size

        mapped = mapped.astype(int).map(mapping)

        if (size < support.size) and (not self.zeros_only):
            newdom += 1

        if not as_df:
            domain[col] = newdom
            df[col] = mapped
            return Dataset(df=df, domain=Domain.fromdict(domain))

        return mapped, newdom

    def transform(self, data):
        as_df = isinstance(data, pd.DataFrame)
        if as_df:
            df = data.copy()
        else:
            df = data.df.copy()

        newdom = {}
        for col in df.columns:
            df[col], newdom[col] = self.transform_col(df, col)
        newdom = Domain.fromdict(newdom)
        if as_df:
            data = df
        else:
            data = Dataset(df, newdom)
        return data

    def reverse(self, data):
        df = data.df.copy()
        newdom = {}
        for col in data.domain:
            support = self.supports[col]
            if self.zeros_only:
                support_map = dict(
                    zip(np.arange(support.sum()), np.arange(support.size)[support])
                )
                df[col] = df[col].map(support_map)
                newdom[col] = support.size
            else:
                mx = support.sum()
                newdom[col] = int(support.size)
                agg = ~support
                if col in self.zeros:
                    agg &= ~self.zeros[col]
                idx, extra = np.where(support)[0], np.where(agg)[0]
                mask = df[col] == mx
                if extra.sum() == 0:
                    extra = idx

                df.loc[mask, col] = self.prng.choice(extra, mask.sum())
                df.loc[~mask, col] = idx[df.loc[~mask, col]]
        newdom = Domain.fromdict(newdom)
        return Dataset(df, newdom)
