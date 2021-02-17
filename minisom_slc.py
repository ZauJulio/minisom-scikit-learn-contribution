"""
    The :mod:`minisom_slc.Minisom_slc` contains the interface for the implementation
    of the Self-Organizing Maps.
"""

# Author: Zaú Júlio <zauhdf@gmail.com>
# License: BSD 3 clause & Creative Commons Attribution 3.0
# Notes: MiniSom by Giuseppe Vettigli is licensed under the Creative Commons Attribution 3.0
#        Unported License. To view a copy of this license, visit
#        http://creativecommons.org/licenses/by/3.0/.

from numpy import argmin, array, unique
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn import svm

from minisom import MiniSom, asymptotic_decay, _build_iteration_indexes


class Minisom_slc(BaseEstimator, ClusterMixin, MiniSom):
    """Unsupervised clusterer estimator."""
    _estimator_type = "clusterer"

    def __init__(self, n_cols, n_rows, input_len, num_iteration, sigma=1.0, learning_rate=0.5,
                 decay_function=asymptotic_decay,
                 neighborhood_function='gaussian', topology='rectangular',
                 activation_distance='euclidean', random_seed=None,
                 random_order=False, verbose=False):
        """Initializes a Self Organizing Maps.

        A rule of thumb to set the size of the grid for a dimensionality
        reduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.

        E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
        hence a map 8-by-8 should perform well.

        Parameters
        ----------
        x : int
            x dimension of the SOM.

        y : int
            y dimension of the SOM.

        input_len : int
            Number of the elements of the vectors in input.

        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
        learning_rate : initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)

        decay_function : function (default=None)
            Function that reduces learning_rate and sigma at each iteration
            the default function is:
                        learning_rate / (1+t/(max_iterarations/2))

            A custom decay function will need to to take in input
            three parameters in the following order:

            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed


            Note that if a lambda function is used to define the decay
            MiniSom will not be pickable anymore.

        neighborhood_function : string, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map.
            Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'

        topology : string, optional (default='rectangular')
            Topology of the map.
            Possible values: 'rectangular', 'hexagonal'

        activation_distance : string, optional (default='euclidean')
            Distance used to activate the map.
            Possible values: 'euclidean', 'cosine', 'manhattan', 'chebyshev'

        random_seed : int, optional (default=None)
            Random seed to use.
        """
        super.__init__(self, n_cols, n_rows, input_len, sigma=1.0, learning_rate=0.5,
                       decay_function=asymptotic_decay,
                       neighborhood_function='gaussian', topology='rectangular',
                       activation_distance='euclidean', random_seed=None)

        self._check_iteration_number(self.num_iteration)
        self.num_iteration = num_iteration
        self.random_order = random_order
        self.verbose = verbose

    def fit(self, X, y=None):
        """Compute of SOM weights.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        self._check_input_len(X)

        random_generator = None
        if self.random_order:
            random_generator = self._random_generator

        iterations = _build_iteration_indexes(
            len(X),
            self.num_iteration,
            self.verbose,
            random_generator
        )
        
        for t, iteration in enumerate(iterations):
            self.update(
                X[iteration],
                self.winner(X[iteration]),
                t,
                self.num_iteration
            )

        if self.verbose:
            print('\n quantization error:', self.quantization_error(X))

        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = None

        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_'])

        # Input validation
        X = check_array(X)

        closest = argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]
