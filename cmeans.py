"""c-means clustering"""

# Authors: Lucas Simões <c.lucas.simoes@gmail.com>
#
# License: BSD 3 clause


import warnings
import time
import numpy as np
from sklearn.utils import check_array, check_random_state


def _init_u(X, n_clusters, random_state=0):
    """
    Initialize the membership matrix U with random membership grades

    Parameters
    ----------
    X: array, shape (n_samples, n_features)
        The data to classify

    n_clusters: int
        The number of clusters

    random_state : int, RandomState instance
        The generator used to initialize the membership matrix. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    A random pertinence matrix U for X
    """
    X = check_array(X)
    num_samples = X.shape[0]
    random_state = check_random_state(random_state)
    U = random_state.dirichlet(np.ones(n_clusters), size=num_samples)
    return U


def _calculate_c(X, U, m=2):
    """
    Compute the centroid for each cluster

    Parameters
    ----------
    X: array, shape (n_samples, n_features)
        The data to classify

    U: array
        The pertinence matrix U

    m: int >=1
        The fuzzyness coefficient. The higher it is, the fuzzyier the clusters will be.

    Returns
    -------
    The centroid matrix C
    """
    u_power = np.power(U, m).T
    return np.dot(u_power, X) / u_power.sum(axis=1)


def _calculate_u(X, n_clusters, C, m=2):
    """
    Calculates the membership function for step k

    Parameters
    ----------
    X: array, shape (n_samples, n_features)
        The data to classify

    n_clusters: int
        The number of clusters

    C: array
        The centroid matrix

    m: int >=1
        The fuzzyness coefficient. The higher it is, the fuzzyier the clusters will be.
    
    Returns
    -------
    The new membership function for X, given C
    """
    power_coef = 1/(m-1)

    c_aux = C.reshape(1, -1)

    a = 1/np.power(
        np.apply_along_axis(np.linalg.norm, 1,
                            (np.repeat(X, n_clusters, axis=0).reshape(len(X), c_aux.size) - c_aux).reshape(-1, n_clusters)).reshape(-1, n_clusters), 2)
    
    b = a.sum(axis=1)
    return np.power(a / np.repeat(b, n_clusters).reshape(len(b), n_clusters), power_coef)


def cmeans_single(X, n_clusters, m=2, tol=1e-3, max_iter=2000, random_state=None,
           verbose=False, return_n_iter=False,):
    """A single run of c-means clustering algorithm;
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    
    n_clusters : int
        The number of clusters to form as well as the number of 
        centroids to generate.

    m: int >=1
        The fuzzyness coefficient. The higher it is, the fuzzyir the clusters will be.
    
    max_iter: int, optional, default 2000
        Maximum number of iterations of the c-means algorithm to run.

    tol : float, optional
        The relative increment in the results before declaring convergence.

    verbose: boolean, optional
        Verbosity mode.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for membership grades initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    
    return_n_iter : bool, optional
        Whether or not to return the number of iterations.

    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of c-means.
    
    membership: float ndarray 
        Membership U with the final membership grades for each data point in X
    """
    # TODO check each and every parameter used
    if m < 1:
        raise ValueError("Invalid fuzzyness coefficient m"
                         "m=%d must be bigger than one." % m)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    X = np.array(X)
    u_k_minus = _init_u(X, n_clusters, random_state)

    C = _calculate_c(X, u_k_minus, m)
    u_k = _calculate_u(X, n_clusters, C, m)
    k = 1

    steps = 0
    error = 10000
    while(error > tol and steps < max_iter):
        time_s = time.time()
        C = _calculate_c(X, u_k, m)
        u_k_minus = u_k.copy()
        u_k = _calculate_u(X, n_clusters, C, m)

        k += 1
        steps += 1
        error = np.absolute(u_k_minus - u_k).sum()

        if verbose:
            print("step: ", steps, "error", error, "time", time_s - time.time())

    if verbose:
        print("u_check", u_k.sum(axis=1))

    if return_n_iter:
        return C, u_k, steps

    return C, u_k
