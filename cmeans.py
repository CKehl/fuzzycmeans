"""
Implements the fuzzy c-means clustering algorithm
@author Lucas Simões
"""
import time
import numpy as np


def init_u(data, n_c):
    """
    @description Initialize the membership matrix U
    @param n_c: num of cluster centers
    @returns: a random pertinence matrix U for data
    """
    num_samples = data.shape[0]
    U = np.random.dirichlet(np.ones(n_c), size=num_samples)
    return U

def calculate_c(data, U, m):
    """
    @description calculates the clusters centers
    U: membership matrix
    c: number of clusters
    m: fuzzyness coefficient
    """
    u_power = np.power(U, m).T # TODO maybe alocate memory for this calculation
    return np.dot(u_power, data) / u_power.sum(axis=1)

def calculate_u(data, C, m):
    """
    Calculates the membership function for step k
    """
    c = len(C)
    power_coef = 1/(m-1)
    a = 1/np.power(np.abs(np.repeat(data, c).reshape(len(data), c) - C), 2)
    b = a.sum(axis=1)
    return np.power(a / np.repeat(b, c).reshape(len(b), c), power_coef)
    


def cmeans(data, n_c, m, epsilon, max_steps=2000, verbose=False):
    """
    Run the C-Means algorithm in @data with @c_n clusters;
    
    The fuzzifier m determines the level of cluster fuzziness.
    A large  m results in smaller membership values, w_{ij} w_{ij}, 
    and hence, fuzzier clusters. In the limit m=1 , the memberships,  w_{ij} w_{ij}, 
    converge to 0 or 1, which implies a crisp partitioning. In the absence of 
    experimentation or domain knowledge, m is commonly set to 2. The algorithm 
    minimizes intra-cluster variance as well, but has the same problems as 
    k-means; the minimum is a local minimum, and the results depend on the initial 
    choice of weights. 

    e: epsilon. Finish condition 
    m: fuzzyness coefficient
    Returns the Membership function U and cluter centers
    """
    if(verbose):
        print("receive data with length", len(data), "c", n_c, "m", m)
    data = np.array(data)
    U_k_minus = init_u(data, n_c)
    
    C = calculate_c(data, U_k_minus, m)
    U_k = calculate_u(data, C, m)
    k = 1

    steps = 0
    error = 10000
    while(error > epsilon and steps < max_steps):
        time_s = time.time()
        C = calculate_c(data, U_k, m)
        U_k_minus = U_k.copy()
        U_k = calculate_u(data, C, m)

        k += 1
        steps += 1
        error = np.absolute(U_k_minus - U_k).sum()

        if verbose:
            print("step: ", steps, "error", error, "time", time_s - time.time())

    if verbose:
        print("u_check", U_k.sum(axis=1))
    return C, U_k

