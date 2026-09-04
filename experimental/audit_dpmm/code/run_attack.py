import string
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool, cpu_count

from mst import MST


N_ROWS = 10
N_COLS = 3
N_ALL = 5000
LEN_SYNTH = 25

EPSILON = 1
DELTA = 1e-2


def featurize_df_queries(df, queries):
    features = np.zeros(len(queries))
    for i, query in enumerate(queries):
        features[i] = (df == query).all(axis=1).sum()
    return features.astype(int)


def featurize_model(model, columns):
    meas = model.measures

    measures = np.zeros(2 * len(columns))
    for col_idx, col in enumerate(columns):
        col_proj = sorted([_meas for _meas in meas if col in _meas[3]], key=lambda x: len(x[3]))

        proj = col_proj[0][3]
        _meas = col_proj[0][1]
        _meas = _meas.reshape(*[_meas.size // 2**(len(proj) - 1) for _ in proj])

        if len(col_proj[0][3]) > 1:
            axis = col_proj[0][3].index(col)
            _meas = np.sum(_meas, axis=tuple([i for i in range(len(_meas.shape)) if i != axis]))

        measures[2 * col_idx: (2 * col_idx) + _meas.shape[0]] = _meas

    return measures


def one_iteration(args):
    i, df_out, df_in, columns, domain, queries, epsilon, delta, len_synth = args

    # out data
    gen_out = MST(epsilon=epsilon, delta=delta, domain=domain, compress=False, n_jobs=1)
    gen_out.fit(df_out)
    synth_out = gen_out.generate(len_synth)
    out_feats = np.concatenate([featurize_df_queries(synth_out, queries), featurize_model(gen_out, columns)])

    # in data
    gen_in = MST(epsilon=epsilon, delta=delta, domain=domain, compress=False, n_jobs=1)
    gen_in.fit(df_in)
    synth_in = gen_in.generate(len_synth)
    in_feats = np.concatenate([featurize_df_queries(synth_in, queries), featurize_model(gen_in, columns)])

    return i, out_feats, in_feats


if __name__ == "__main__":
    # data
    columns = list(string.ascii_uppercase[:N_COLS])
    domain = {col: 2 for col in columns}

    df_out = pd.DataFrame(np.zeros((N_ROWS, N_COLS), dtype=int), columns=columns)
    df_in = pd.DataFrame(np.vstack([np.ones((1, N_COLS), dtype=int), np.zeros((N_ROWS, N_COLS), dtype=int)]), columns=columns)

    # black-box + white-box features
    queries = np.array(list(product([0, 1], repeat=N_COLS)))
    n_features = len(queries) + 2 * len(columns)
    data = {"out": np.zeros([N_ALL, n_features]), "in": np.zeros([N_ALL, n_features])}

    # build tasks
    tasks = [(i, df_out, df_in, columns, domain, queries, EPSILON, DELTA, LEN_SYNTH) for i in range(N_ALL)]
    n_cpu = max(1, cpu_count() - 1)

    with Pool(processes=n_cpu, maxtasksperchild=1) as pool:
        for i, out_row, in_row in tqdm(
            pool.imap_unordered(one_iteration, tasks, chunksize=1),
            total=N_ALL,
            desc="it",
            leave=False,
        ):
            data["out"][i, :] = out_row
            data["in"][i, :] = in_row

    with open('../data/features.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
