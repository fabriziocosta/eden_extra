#!/usr/bin/env python

"""Cluster is a cluster algorithm.

@author: Fabrizio Costa
@email: costa@informatik.uni-freiburg.de
"""

import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cross_validation import cross_val_predict

import logging
logger = logging.getLogger(__name__)


class RandomForestClassifierWrapper(RandomForestClassifier):
    """Replace predict with predict_proba."""

    def predict(self, data_matrix):
        """predict."""
        return RandomForestClassifier.predict_proba(self, data_matrix)


class QuickShiftClusterer(object):
    """Clustering based on contraction/expansion of nearest neighbors.

    An initial clustering is performed. The clusters are used as supervised
    target information. A random forest is trained to consistently discriminate
    between the clusters. The confidence of the cross validated prediction
    is computed.
    The knn are computed.
    The k shift neighbors are computed: a k shift neighbor is the k nearest
    denser neighbor that is most likely to have a different class.
    The density of instance A is defined as the average pairwise cosine
    similarity of A wrt all other instances.
    Outlier nodes are not detached. Outlier nodes are defined as those
    instances that have no mutual
    k=mutual_knn neighbors.

    """

    def __init__(self,
                 k=5,
                 factor=0.1,
                 n_iter=3,
                 similarity_th=.5,
                 n_clusters=2,
                 clusterer=SpectralClustering(),
                 random_state=1,
                 metric='cosine', **kwds):
        """Constructor."""
        self.__version__ = '1.0.1'
        self.random_state = random_state
        random.seed(random_state)
        self.k = k
        self.factor = factor
        self.n_iter = n_iter
        self.similarity_th = similarity_th
        self.n_clusters = n_clusters
        self.clusterer = clusterer
        self.classifier = RandomForestClassifierWrapper()
        self.metric = metric
        self.kwds = kwds

    def predict(self, data_matrix):
        """predict."""
        local_data_matrix = minmax_scale(data_matrix)

        # main iteration
        for iteration in range(self.n_iter):
            # cluster
            self.clusterer.set_params(n_clusters=self.n_clusters)
            y = self.clusterer.fit_predict(local_data_matrix)
            # compute confidence margin
            probs = cross_val_predict(
                self.classifier, local_data_matrix, y, cv=3)
            # compute all links matrices
            a, b, c = self.compute_matrix(data_matrix=local_data_matrix,
                                          probs=probs,
                                          similarity_th=self.similarity_th,
                                          k=self.k)
            knns_matrix, kshift_matrix, mutual_knn_counts = a, b, c
            # perform in place expansion and contraction
            local_data_matrix = self.contract_expand(
                data_matrix=local_data_matrix,
                knns_matrix=knns_matrix,
                kshift_matrix=kshift_matrix,
                mutual_knn_counts=mutual_knn_counts)

        self.data_matrix = local_data_matrix
        return y

    def contract_expand(self,
                        data_matrix=None,
                        knns_matrix=None,
                        kshift_matrix=None,
                        mutual_knn_counts=None):
        """contract_expand.

        knn instances are contracted i.e. they become closer to each other.
        shift links expand distances.
        """
        # move outliers towards centers of their neighbors
        for i, count in enumerate(mutual_knn_counts):
            i = int(i)
            x = data_matrix[i]
            if count == 0:
                z = np.mean(knns_matrix[i], axis=0)
                step = (z - x) * float(self.factor)
                x = x + step
                data_matrix[i] = x

        # shift links expand distances i.e. they become further apart from
        # each other; unless they are outliers, i.e. they have 0 mutual
        # k nearest neighbors
        for i, links in enumerate(kshift_matrix):
            i = int(i)
            x = data_matrix[i]
            for rank, j in enumerate(links):
                j = int(j)
                z = data_matrix[j]
                step = (z - x) * float(self.factor)
                x = x - step
                z = z + step
                data_matrix[i] = x
                data_matrix[j] = z

        # knn instances are contracted i.e. they become closer to
        # each other.
        for i, knns in enumerate(knns_matrix):
            i = int(i)
            x = data_matrix[i]
            for rank, j in enumerate(knns):
                j = int(j)
                z = data_matrix[j]
                step = (z - x) * float(self.factor)
                x = x + step
                z = z - step
                data_matrix[i] = x
                data_matrix[j] = z
        # rescale data to unit hyper cube in positive orthant
        data_matrix = minmax_scale(data_matrix)
        return data_matrix

    def compute_matrix(self, data_matrix=None,
                       probs=None,
                       similarity_th=None,
                       k=None):
        """compute_matrix."""
        size = data_matrix.shape[0]
        # make kernel
        kernel_matrix = pairwise_kernels(data_matrix,
                                         metric=self.metric,
                                         **self.kwds)
        # compute instance density as average pairwise similarity
        density = np.sum(kernel_matrix, 0) / size
        # compute list of nearest neighbors
        kernel_matrix_sorted_ids = np.argsort(-kernel_matrix)
        # make matrix of densities ordered by nearest neighbor
        density_matrix = density[kernel_matrix_sorted_ids]

        knns_matrix = self._knns(
            k=k,
            probs=probs,
            similarity_th=similarity_th,
            kernel_matrix=kernel_matrix,
            density_matrix=density_matrix,
            kernel_matrix_sorted_ids=kernel_matrix_sorted_ids)

        # build shift links
        kshift_matrix = self._kshifts(
            k=k,
            probs=probs,
            similarity_th=similarity_th,
            kernel_matrix=kernel_matrix,
            density_matrix=density_matrix,
            kernel_matrix_sorted_ids=kernel_matrix_sorted_ids)

        # build mutual knn
        mutual_knn_counts = self._mutual_knns(
            k=k,
            kernel_matrix=kernel_matrix,
            kernel_matrix_sorted_ids=kernel_matrix_sorted_ids)

        return knns_matrix, kshift_matrix, mutual_knn_counts

    def _set_max_similarity(self, instance, instances):
        return max([instance.dot(i) for i in instances])

    def _set_min_similarity(self, instance, instances):
        return min([instance.dot(i) for i in instances])

    def _knns(self,
              k=None,
              probs=None,
              similarity_th=1,
              kernel_matrix=None,
              density_matrix=None,
              kernel_matrix_sorted_ids=None):
        """_mutual_knn."""
        size = kernel_matrix.shape[0]
        knns_matrix = np.zeros((size, k))
        # for all instances determine link
        for i in range(size):
            knn_instances = []
            knn_instances.append(probs[i])
            counter = 0
            # add edges to the knns
            for jj in range(1, size):
                j = kernel_matrix_sorted_ids[i, jj]
                # if the similarity is high wrt
                # previous selections
                similarity = self._set_min_similarity(
                    probs[j], knn_instances)
                if similarity > similarity_th:
                    knns_matrix[i, counter] = j
                    knn_instances.append(probs[j])
                    counter += 1
                # proceed until counter reaches k
                if counter == k:
                    break
        return knns_matrix

    def _kshifts(self,
                 k=None,
                 probs=None,
                 similarity_th=1,
                 kernel_matrix=None,
                 density_matrix=None,
                 kernel_matrix_sorted_ids=None):
        size = kernel_matrix.shape[0]
        kshift_matrix = np.zeros((size, k))
        # for all instances determine link
        for i, densities in enumerate(density_matrix):
            k_shift_instances = []
            k_shift_instances.append(probs[i])
            # if a denser neighbor cannot be found then assign link to the
            # instance itself
            kshift_matrix[i] = np.array([i] * k)
            i_density = densities[0]
            counter = 0
            # for all neighbors from the closest to the furthest
            for jj, j_density in enumerate(densities):
                if jj > 0:
                    j = kernel_matrix_sorted_ids[i, jj]
                    # if the density of the neighbor is higher
                    if j_density > i_density:
                        # if the similarity is not too high wrt
                        # previous selections
                        similarity = self._set_max_similarity(
                            probs[j], k_shift_instances)
                        if similarity < similarity_th:
                            kshift_matrix[i, counter] = j
                            k_shift_instances.append(probs[j])
                            counter += 1
                # proceed until counter reaches k
                if counter == k:
                    break
        return kshift_matrix

    def _mutual_knns(self,
                     k=None,
                     kernel_matrix=None,
                     kernel_matrix_sorted_ids=None):
        """_mutual_knn."""
        size = kernel_matrix.shape[0]
        mutual_knn_counts = np.zeros(size)
        for i in range(size):
            counter = 0
            # add edges to the knns
            for jj in range(1, int(k) + 1):
                j = kernel_matrix_sorted_ids[i, jj]
                if i != j:
                    # check that within the k-nn also i is a knn of j
                    # i.e. use the symmetric nneighbor notion
                    upto = int(k) + 1
                    i_knns = kernel_matrix_sorted_ids[j, :upto]
                    if i in list(i_knns):
                        counter += 1
            mutual_knn_counts[i] = counter
        return mutual_knn_counts
