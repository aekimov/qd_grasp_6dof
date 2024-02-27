

import numpy as np
from sklearn.neighbors import NearestNeighbors as Nearest
from scipy.spatial import cKDTree as KDTree

INF_NN_DIST = 1000000000  # value for infinity distance for novelty computation
K_NN_NOV = 15  # number of nearest neighbours for novelty computation


def compute_average_distance_array(query, k_tree):
    """Finds K nearest neighbours and distances

    Args:
        query (List): behavioral descriptor of individual
        k_tree (Nearest): tree in the behavior descriptor space

    Returns:
        float: average distance to the K nearest neighbours
    """

    n_samples = k_tree.n_samples_fit_
    query = np.array(query)
    if n_samples >= K_NN_NOV + 1:
        neighbours_distances = k_tree.kneighbors(X=query)[0][:, 1:]
    else:
        neighbours_distances = k_tree.kneighbors(X=query, n_neighbors=n_samples)[0][:, 1:]

    avg_distances = np.mean(neighbours_distances, axis=1)
    return avg_distances


def assess_novelties_single_bd_vec(pop_bds, reference_pop_bds):
    k_tree = Nearest(n_neighbors=K_NN_NOV + 1, metric='minkowski')
    k_tree.fit(reference_pop_bds)
    novelties = compute_average_distance_array(pop_bds, k_tree)
    return novelties


def compute_average_distance(query, k_tree, expected_neighbours=False):
    """Finds K nearest neighbours and distances

    Args:
        query (List): behavioral descriptor of individual
        k_tree (KDTree or Nearest): tree in the behavior descriptor space

    Returns:
        float: average distance to the K nearest neighbours
        list: indices of the K nearest neighbours
    """
    is_query_invalid = None in query
    if is_query_invalid:
        return (None, None) if expected_neighbours else (None,)  # force consistacy with the code

    if isinstance(k_tree, KDTree):
        raise AttributeError('Depreciated type. Func is still in the code for legacy purpose. Check if necessary.')

    elif isinstance(k_tree, Nearest):
        avg_distance, neighbours_indices = compute_average_distance_nearest(query, k_tree)
    else:
        raise AttributeError(f'Invalid k_tree type: {type(k_tree)} (supported : Nearest)')

    return avg_distance, neighbours_indices


# for debug profiling
def fit_bd_neareast_neighbours_map(tuple_args):
    """Fit kd tree (kNN algo using nov_metric metric) for the given lise of behavior descriptors. Returns the fitted
    kdtree."""
    bd_list, nov_metric = tuple_args
    if len(bd_list) == 0:
        return None

    neigh = Nearest(n_neighbors=K_NN_NOV + 1, metric=nov_metric)
    bds_idx_arr = np.array(bd_list)
    neigh.fit(bds_idx_arr)
    return neigh


def compute_average_distance_nearest(query, k_tree):

    n_samples = k_tree.n_samples_fit_
    query = np.array(query)

    n_neigh = K_NN_NOV if n_samples >= K_NN_NOV + 1 else n_samples

    search = k_tree.kneighbors(X=query.reshape(1, -1), n_neighbors=n_neigh)

    neighbours_distances = search[0][0][1:]
    neighbours_indices = search[1][0][1:]

    avg_distance = np.mean(neighbours_distances).item() if len(neighbours_distances) != 0 else INF_NN_DIST

    return avg_distance, neighbours_indices


def assess_novelties(pop, archive):
    """Compute novelties of current population."""

    if archive:
        reference_pop_bds = np.concatenate((pop.bds, archive.bds), axis=0)
    else:
        reference_pop_bds = pop.bds

    novelties = assess_novelties_single_bd_vec(
        pop_bds=pop.bds,
        reference_pop_bds=reference_pop_bds
    )

    return novelties


