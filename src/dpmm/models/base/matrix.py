# This file is part of the Ektelo framework
# For licensing terms please see: https://github.com/ektelo/ektelo
#
# Copyright 2019-2021, Tumult Labs Inc.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

# from functools import reduce
# import math


class EkteloMatrix(LinearOperator):
    """
    An EkteloMatrix is a linear transformation that can compute matrix-vector products
    """

    # must implement: _matmat, _transpose, matrix
    # can  implement: gram, sensitivity, sum, dense_matrix, sparse_matrix, __abs__

    def __init__(self, matrix):
        """Instantiate an EkteloMatrix from an explicitly represented backing matrix

        :param matrix: a 2d numpy array or a scipy sparse matrix
        """
        self.matrix = matrix
        self.dtype = matrix.dtype
        self.shape = matrix.shape

    def _transpose(self):
        return EkteloMatrix(self.matrix.T)

    def _matmat(self, v_arr):
        """
        Matrix multiplication of a m x n matrix Q

        :param V: a n x p numpy array
        :return Q*V: a m x p numpy aray
        """
        return self.matrix @ v_arr

    def gram(self):
        """
        Compute the Gram matrix of the given matrix.
        For a matrix Q, the gram matrix is defined as Q^T Q
        """
        return self.T @ self  # works for subclasses too

    def sensitivity(self):
        # note: this works because np.abs calls self.__abs__
        return np.max(np.abs(self).sum(axis=0))

    def sum(self, axis=None):
        # this implementation works for all subclasses too
        # (as long as they define _matmat and _transpose)
        if axis == 0:
            return self.T.dot(np.ones(self.shape[0]))
        ans = self.dot(np.ones(self.shape[1]))
        return ans if axis == 1 else np.sum(ans)

    def inv(self):
        return EkteloMatrix(np.linalg.inv(self.dense_matrix()))

    def pinv(self):
        return EkteloMatrix(np.linalg.pinv(self.dense_matrix()))

    def trace(self):
        return self.diag().sum()

    def diag(self):
        return np.diag(self.dense_matrix())

    def _adjoint(self):
        return self._transpose()

    def __mul__(self, other):
        if np.isscalar(other):
            return Weighted(self, other)
        if type(other) is np.ndarray:
            return self.dot(other)
        if isinstance(other, EkteloMatrix):
            return Product(self, other)
            # note: this expects both matrix types to be compatible (e.g., sparse and sparse)
            # todo: make it work for different backing representations
        else:
            raise TypeError(
                "incompatible type %s for multiplication with EkteloMatrix"
                % type(other)
            )

    def __add__(self, other):
        if np.isscalar(other):
            other = Weighted(Ones(self.shape), other)
        return Sum([self, other])

    def __sub__(self, other):
        return self + -1 * other

    def __rmul__(self, other):
        if np.isscalar(other):
            return Weighted(self, other)
        return NotImplemented

    def __getitem__(self, key):
        """
        return a given row from the matrix

        :param key: the index of the row to return
        :return: a 1xN EkteloMatrix
        """
        # row indexing, subclasses may provide more efficient implementation
        m, n = self.shape
        v = np.zeros(m)
        v[key] = 1.0
        return EkteloMatrix(self.T.dot(v).reshape(1, n))

    def dense_matrix(self):
        """
        return the dense representation of this matrix, as a 2D numpy array
        """
        if sparse.issparse(self.matrix):
            return self.matrix.toarray()
        return self.matrix

    def sparse_matrix(self):
        """
        return the sparse representation of this matrix, as a scipy matrix
        """
        if sparse.issparse(self.matrix):
            return self.matrix
        return sparse.csr_matrix(self.matrix)

    @property
    def ndim(self):
        # todo: deprecate if possible
        return 2

    def __abs__(self):
        return EkteloMatrix(self.matrix.__abs__())

    def __sqr__(self):
        if sparse.issparse(self.matrix):
            return EkteloMatrix(self.matrix.power(2))
        return EkteloMatrix(self.matrix**2)

    def l1_sensitivity(self):
        return self.__abs__().sum(axis=0).max()

    def l2_sensitivity(self):
        return np.sqrt(self.__sqr__().sum(axis=0).max())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self):
        return hash(repr(self))


class Identity(EkteloMatrix):
    def __init__(self, n, dtype=np.float64):
        self.n = n
        self.shape = (n, n)
        self.dtype = dtype

    def _matmat(self, v_arr):
        return v_arr

    def _transpose(self):
        return self

    @property
    def matrix(self):
        return sparse.eye(self.n, dtype=self.dtype)

    def __mul__(self, other):
        assert other.shape[0] == self.n, "dimension mismatch"
        return other

    def inv(self):
        return self

    def pinv(self):
        return self

    def trace(self):
        return self.n

    def __abs__(self):
        return self

    def __sqr__(self):
        return self


class Ones(EkteloMatrix):
    """A m x n matrix of all ones"""

    def __init__(self, m, n, dtype=np.float64):
        self.m = m
        self.n = n
        self.shape = (m, n)
        self.dtype = dtype

    def _matmat(self, v_arr):
        ans = v_arr.sum(axis=0, keepdims=True)
        return np.repeat(ans, self.m, axis=0)

    def _transpose(self):
        return Ones(self.n, self.m, self.dtype)

    def gram(self):
        return self.m * Ones(self.n, self.n, self.dtype)

    def pinv(self):
        c = 1.0 / (self.m * self.n)
        return c * Ones(self.n, self.m, self.dtype)

    def trace(self):
        assert self.n == self.m, "matrix is not square"
        return self.n

    @property
    def matrix(self):
        return np.ones(self.shape, dtype=self.dtype)

    def __abs__(self):
        return self

    def __sqr__(self):
        return self


class Weighted(EkteloMatrix):
    """Class for multiplication by a constant"""

    def __init__(self, base, weight):
        if isinstance(base, Weighted):
            weight *= base.weight
            base = base.base
        self.base = base
        self.weight = weight
        self.shape = base.shape
        self.dtype = base.dtype

    def _matmat(self, v_arr):
        return self.weight * self.base.dot(v_arr)

    def __mul__(self, other):
        if isinstance(other, EkteloMatrix):
            return Weighted(self.base @ other, self.weight)
        return EkteloMatrix.__mul__(self, other)

    def _transpose(self):
        return Weighted(self.base.T, self.weight)

    def gram(self):
        return Weighted(self.base.gram(), self.weight**2)

    def pinv(self):
        return Weighted(self.base.pinv(), 1.0 / self.weight)

    def inv(self):
        return Weighted(self.base.inv(), 1.0 / self.weight)

    def trace(self):
        return self.weight * self.base.trace()

    def __abs__(self):
        return Weighted(self.base.__abs__(), np.abs(self.weight))

    def __sqr__(self):
        return Weighted(self.base.__sqr__(), self.weight**2)

    def sensitivity(self):
        return self.weight * self.base.sensitivity()

    @property
    def matrix(self):
        return self.weight * self.base.matrix


class Sum(EkteloMatrix):
    """Class for the Sum of matrices"""

    def __init__(self, matrices):
        # all must have same shape
        self.matrices = matrices
        self.shape = matrices[0].shape
        self.dtype = np.result_type(*[q_matrix.dtype for q_matrix in matrices])

    def _matmat(self, v_arr):
        return sum(q_matrix.dot(v_arr) for q_matrix in self.matrices)

    def _transpose(self):
        return Sum([q_matrix.T for q_matrix in self.matrices])

    def __mul__(self, other):
        if isinstance(other, EkteloMatrix):
            return Sum(
                [q_matrix @ other for q_matrix in self.matrices]
            )  # should use others rmul though
        return EkteloMatrix.__mul__(self, other)

    def diag(self):
        return sum(q_matrix.diag() for q_matrix in self.matrices)

    def trace(self):
        return sum(q_matrix.trace() for q_matrix in self.matrices)

    @property
    def matrix(self):
        if _any_sparse(self.matrices):
            return sum(q_matrix.sparse_matrix() for q_matrix in self.matrices)
        return sum(q_matrix.dense_matrix() for q_matrix in self.matrices)


class Product(EkteloMatrix):
    def __init__(self, a, b):
        assert a.shape[1] == b.shape[0]
        self._A = a
        self._B = b
        self.shape = (a.shape[0], b.shape[1])
        self.dtype = np.result_type(a.dtype, b.dtype)

    def _matmat(self, x_arr):
        return self._A.dot(self._B.dot(x_arr))

    def _transpose(self):
        return Product(self._B.T, self._A.T)

    @property
    def matrix(self):
        return self._A.matrix @ self._B.matrix

    def gram(self):
        return Product(self.T, self)

    def inv(self):
        return Product(self._B.inv(), self._A.inv())


def _any_sparse(matrices):
    return any(sparse.issparse(q_matrix.matrix) for q_matrix in matrices)
