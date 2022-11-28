import pickle
import struct
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import json
plt.rcParams['text.usetex'] = True
import time
from datasketch.storage import (
    ordered_storage, unordered_storage, _random_name)

from scipy.integrate import quad as integrate

import logging
log_format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(filename="logs.txt",
                    filemode="w",
                    format=log_format,
                    level=logging.INFO)

logger = logging.getLogger('minhash')


def _false_positive_probability(threshold, b_1, b_2, r_1, r_2):
    _probability = lambda s : 1 - (1 - (1 - (1 - s**float(r_1))**float(b_1))**float(r_2))**float(b_2)
    a, err = integrate(_probability, 0.0, threshold)
    return a


def _false_negative_probability(threshold, b_1, b_2, r_1, r_2):
    _probability = lambda s : 1 - (1 - (1 - (1 - (1 - s**float(r_1))**float(b_1))**float(r_2))**float(b_2))
    a, err = integrate(_probability, threshold, 1.0)
    return a


def _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight, amplified):
    '''
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    '''
    min_error = float("inf")
    opt = ((), ())
    count = 0
    for b_0 in range(1, num_perm + 1):
        r_1 = int(num_perm / b_0)
        b_1_start_range = 1 if amplified else b_0
        for b_1 in range(b_1_start_range, b_0 + 1):
            n_2 = int(b_0 / b_1)
            for b_2 in range(1, n_2 + 1):
                count += 1
                r_2 = int(n_2 / b_2)
                fp = _false_positive_probability(threshold, b_1, b_2, r_1, r_2)
                fn = _false_negative_probability(threshold, b_1, b_2, r_1, r_2)
                error = fp*false_positive_weight + fn*false_negative_weight
                # if count % 1000 == 0:
                #     logger.info('params: {}'.format((f'b_0:{b_0}', f'b_1:{b_1}', f'b_2:{b_2}', f'r_1:{r_1}', f'r_2:{r_2}')))
                #     logger.info('error score: {}'.format(error))
                if error < min_error:
                    min_error = error
                    b = (b_1, b_2)
                    r = (r_1, r_2)
                    opt = (b, r)
    print(f'optimal_parameters: {opt}')
    return opt


def manual_get_divisor(number):
    return_set = set()
    for i in range(1, int(number / 2) + 1):
        if number % i == 0:
            return_set.add(i)
            return_set.add(int(number / i) )
    return return_set


def _optimal_param_new(threshold, num_perm, false_positive_weight, false_negative_weight, amplified, minimum_r1=1):
    '''
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    '''
    start_time = time.time()
    min_error = float("inf")
    opt = ((), ())
    # all_outcomes = []
    count = 0
    for r_1 in range(minimum_r1, num_perm + 1):
        print(r_1)
        max_b_0 = int(num_perm / r_1)
        for b_0 in range(max_b_0, 0, -1):
            b_1_options = manual_get_divisor(b_0) if amplified else {b_0}
            for b_1 in b_1_options:
                n_2 = int(b_0 / b_1)
                b_2_options = manual_get_divisor(n_2) if amplified else {1}
                for b_2 in b_2_options:
                    count += 1
                    r_2 = int(n_2 / b_2)
                    fp = _false_positive_probability(threshold, b_1, b_2, r_1, r_2)
                    fn = _false_negative_probability(threshold, b_1, b_2, r_1, r_2)
                    error = fp * false_positive_weight + fn * false_negative_weight

                    if error < min_error:
                        min_error = error
                        b = (b_1, b_2)
                        r = (r_1, r_2)
                        opt = (b, r)
    print(f'optimal_parameters: {opt}')
    print(f'evaluation took {time.time() - start_time} seconds')
    return opt

# TODO
# Create groups of hashtables
# Per group of hashtables, generate candidate pairs.
# If candidate pair is found in all groups that contribute to a specific OR part of the final stage, then it is a real candidate pair

class MinHashLSHAmplified(object):
    '''
    The :ref:`minhash_lsh` index.
    It supports query with `Jaccard similarity`_ threshold.
    Reference: `Chapter 3, Mining of Massive Datasets
    <http://www.mmds.org/>`_.

    Args:
        threshold (float): The Jaccard similarity threshold between 0.0 and
            1.0. The initialized MinHash LSH will be optimized for the threshold by
            minizing the false positive and false negative.
        num_perm (int, optional): The number of permutation functions used
            by the MinHash to be indexed. For weighted MinHash, this
            is the sample size (`sample_size`).
        weights (tuple, optional): Used to adjust the relative importance of
            minimizing false positive and false negative when optimizing
            for the Jaccard similarity threshold.
            `weights` is a tuple in the format of
            :code:`(false_positive_weight, false_negative_weight)`.
        params (tuple, optional): The LSH parameters (i.e., number of bands and size
            of each bands). This is used to bypass the parameter optimization
            step in the constructor. `threshold` and `weights` will be ignored
            if this is given.
        storage_config (dict, optional): Type of storage service to use for storing
            hashtables and keys.
            `basename` is an optional property whose value will be used as the prefix to
            stored keys. If this is not set, a random string will be generated instead. If you
            set this, you will be responsible for ensuring there are no key collisions.
        prepickle (bool, optional): If True, all keys are pickled to bytes before
            insertion. If None, a default value is chosen based on the
            `storage_config`.
        hashfunc (function, optional): If a hash function is provided it will be used to
            compress the index keys to reduce the memory footprint. This could cause a higher
            false positive rate.

    Note:
        `weights` must sum to 1.0, and the format is
        (false positive weight, false negative weight).
        For example, if minimizing false negative (or maintaining high recall) is more
        important, assign more weight toward false negative: weights=(0.4, 0.6).
        Try to live with a small difference between weights (i.e. < 0.5).
    '''

    def __init__(self, threshold=0.9, num_perm=128, weights=(0.5, 0.5), amplified=False,
                 params=None, storage_config=None, prepickle=None, hashfunc=None, parameter_config_list=[], minimum_r1=1):
        storage_config = {'type': 'dict'} if not storage_config else storage_config
        if params is None:
            params_config = [x['params'] for x in parameter_config_list if (abs(x['threshold'] - threshold) < 0.02
                                                                        and x['num_perm'] == num_perm
                                                                        and (x['amplified'] is amplified))]
        self._buffer_size = 50000
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.threshold = threshold
        self.h = num_perm
        if params is not None:
            self.b, self.r = params
            if self.b[0] * self.b[1] * self.r[0] * self.r[1] > num_perm:
                raise ValueError("The product of b_1, b_2, r_1 and r_2 in params is "
                        "{} * {} = {} -- it must be less than num_perm {}. "
                        "Did you forget to specify num_perm?".format(
                            self.b[0], self.b[1], self.r[0], self.r[1], self.b[0] * self.b[1] * self.r[0] * self.r[1], num_perm))
        elif len(params_config) > 0:
            self.b, self.r = next(x['params'] for x in parameter_config_list if (abs(x['threshold'] - threshold) < 0.02
                                                            and x['num_perm'] == num_perm
                                                            and (x['amplified'] is amplified)))
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r = _optimal_param_new(threshold, num_perm, false_positive_weight, false_negative_weight, amplified, minimum_r1)
        self.total_bands = int(self.r[1] * self.b[0] * self.b[1])
        self.prepickle = storage_config['type'] == 'redis' if prepickle is None else prepickle

        self.hashfunc = hashfunc
        if hashfunc:
            self._H = self._hashed_byteswap
        else:
            self._H = self._byteswap

        basename = storage_config.get('basename', _random_name(11))
        self.hashtables = [
            unordered_storage(storage_config, name=b''.join([basename, b'_bucket_', struct.pack('>H', i)]))
            for i in range(self.total_bands)]
        self.hashranges = [(i*self.r[0], (i+1)*self.r[0]) for i in range(self.total_bands)]
        self.keys = ordered_storage(storage_config, name=b''.join([basename, b'_keys']))

    @property
    def buffer_size(self):
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value):
        self.keys.buffer_size = value
        for t in self.hashtables:
            t.buffer_size = value
        self._buffer_size = value

    def insert(self, key, minhash, check_duplication=True):
        '''
        Insert a key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by
        the key.

        :param str key: The identifier of the set.
        :param datasketch.MinHash minhash: The MinHash of the set.
        :param bool check_duplication: To avoid duplicate keys in the storage (`default=True`).
                                       It's recommended to not change the default, but
                                       if you want to avoid the overhead during insert
                                       you can set `check_duplication = False`.
        '''
        self._insert(key, minhash, check_duplication=check_duplication, buffer=False)

    def insertion_session(self, buffer_size=50000):
        '''
        Create a context manager for fast insertion into this index.

        :param int buffer_size: The buffer size for insert_session mode (default=50000).

        Returns:
            datasketch.lsh.MinHashLSHInsertionSession
        '''
        return MinHashLSHInsertionSession(self, buffer_size=buffer_size)

    def _insert(self, key, minhash, check_duplication=True, buffer=False):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                    % (self.h, len(minhash)))
        if self.prepickle:
            key = pickle.dumps(key)
        if check_duplication and key in self.keys:
            raise ValueError("The given key already exists")
        Hs = [self._H(minhash.hashvalues[start:end])
                for start, end in self.hashranges]
        self.keys.insert(key, *Hs, buffer=buffer)
        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H, key, buffer=buffer)

    def query(self, minhash):
        '''
        Giving the MinHash of the query set, retrieve
        the keys that references sets with Jaccard
        similarities greater than the threshold.

        Args:
            minhash (datasketch.MinHash): The MinHash of the query set.

        Returns:
            `list` of unique keys.
        '''
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                    % (self.h, len(minhash)))
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in hashtable.get(H):
                candidates.add(key)
        if self.prepickle:
            return [pickle.loads(key) for key in candidates]
        else:
            return list(candidates)

    def add_to_query_buffer(self, minhash):
        '''
        Giving the MinHash of the query set, buffer
        queries to retrieve the keys that references
        sets with Jaccard similarities greater than
        the threshold.

        Buffered queries can be executed using
        `collect_query_buffer`. The combination of these
        functions is way faster if cassandra backend
        is used with `shared_buffer`.

        Args:
            minhash (datasketch.MinHash): The MinHash of the query set.
        '''
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                             % (self.h, len(minhash)))
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            hashtable.add_to_select_buffer([H])

    def collect_query_buffer(self):
        '''
        Execute and return buffered queries given
        by `add_to_query_buffer`.

        If multiple query MinHash were added to the query buffer,
        the intersection of the results of all query MinHash will be returned.

        Returns:
            `list` of unique keys.
        '''
        collected_result_sets = [
            set(collected_result_lists)
            for hashtable in self.hashtables
            for collected_result_lists in hashtable.collect_select_buffer()
        ]
        if not collected_result_sets:
            return []
        if self.prepickle:
            return [pickle.loads(key) for key in set.intersection(*collected_result_sets)]
        return list(set.intersection(*collected_result_sets))

    def __contains__(self, key):
        '''
        Args:
            key (hashable): The unique identifier of a set.

        Returns:
            bool: True only if the key exists in the index.
        '''
        if self.prepickle:
            key = pickle.dumps(key)
        return key in self.keys

    def remove(self, key):
        '''
        Remove the key from the index.

        Args:
            key (hashable): The unique identifier of a set.

        '''
        if self.prepickle:
            key = pickle.dumps(key)
        if key not in self.keys:
            raise ValueError("The given key does not exist")
        for H, hashtable in zip(self.keys[key], self.hashtables):
            hashtable.remove_val(H, key)
            if not hashtable.get(H):
                hashtable.remove(H)
        self.keys.remove(key)

    def is_empty(self):
        '''
        Returns:
            bool: Check if the index is empty.
        '''
        return any(t.size() == 0 for t in self.hashtables)

    def _byteswap(self, hs):
        return bytes(hs.byteswap().data)

    def _hashed_byteswap(self, hs):
        return self.hashfunc(bytes(hs.byteswap().data))

    def _query_b(self, minhash, b):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                    % (self.h, len(minhash)))
        if b > len(self.hashtables):
            raise ValueError("b must be less or equal to the number of hash tables")
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges[:b], self.hashtables[:b]):
            H = self._H(minhash.hashvalues[start:end])
            if H in hashtable:
                for key in hashtable[H]:
                    candidates.add(key)
        if self.prepickle:
            return {pickle.loads(key) for key in candidates}
        else:
            return candidates

    def get_counts(self):
        '''
        Returns a list of length ``self.b`` with elements representing the
        number of keys stored under each bucket for the given permutation.
        '''
        counts = [
            hashtable.itemcounts() for hashtable in self.hashtables]
        return counts

    def get_subset_counts(self, *keys):
        '''
        Returns the bucket allocation counts (see :func:`~datasketch.MinHashLSH.get_counts` above)
        restricted to the list of keys given.

        Args:
            keys (hashable) : the keys for which to get the bucket allocation
                counts
        '''
        if self.prepickle:
            key_set = [pickle.dumps(key) for key in set(keys)]
        else:
            key_set = list(set(keys))
        hashtables = [unordered_storage({'type': 'dict'}) for _ in
                      range(self.total_bands)]
        Hss = self.keys.getmany(*key_set)
        for key, Hs in zip(key_set, Hss):
            for H, hashtable in zip(Hs, hashtables):
                hashtable.insert(H, key)
        return [hashtable.itemcounts() for hashtable in hashtables]


    def get_candidate_pairs_old(self):
        '''
        Returns a list of tuples of candidate pairs
        '''
        print('retrieving candidate pairs')

        # First produce candidate pairs per group in first layer
        candidates_per_group = {f"group_{i}": set() for i in range(1, int(self.b[1] * self.r[1]) + 1)}
        count = 0
        for index, hash_table in enumerate(self.hashtables):
            current_group = int(index / self.b[0]) + 1
            print(f'current group: {current_group}')
            if current_group > int(self.b[1] * self.r[1]):
                break
            for H in hash_table:
                if len(hash_table[H]) > 1:
                    count += 1
                    pairs = {frozenset({x, y}) for x in hash_table[H] for y in hash_table[H] if x != y}
                    candidates_per_group[f"group_{current_group}"].update(pairs)
        candidate_pairs = set()
        for band_2 in range(1, int(self.b[1]) + 1):
            print(f'band 2: {band_2}')
            # per band in second layer, check if the candidate pair is in every row!
            dict_candidate_pairs = defaultdict(int)
            for row_second_layer in range((band_2 - 1) * self.r[1] + 1, band_2 * self.r[1] + 1):
                print(f'row 2: {row_second_layer}')
                for candidate_pair in candidates_per_group[f"group_{row_second_layer}"]:
                    dict_candidate_pairs[candidate_pair] += 1
            candidate_pairs.update({candidate_pair for candidate_pair in dict_candidate_pairs if dict_candidate_pairs[candidate_pair] == self.r[1]})
        return candidate_pairs


    def get_candidate_pairs(self):
        '''
        Returns a list of tuples of candidate pairs
        '''
        candidate_pairs = set()
        # First produce candidate pairs per group in first layer
        for band_2 in range(1, int(self.b[1]) + 1):
            candidates_per_group = {f"group_{j}": set() for j in range(0, int(self.r[1]))}
            for index, hash_table in enumerate(self.hashtables[(band_2 - 1) * self.r[1] * self.b[0]: band_2 * self.r[1] * self.b[0]]):
                current_group = int(index / self.b[0])
                for H in hash_table:
                    if len(hash_table[H]) > 1:
                        pairs = {frozenset({x, y}) for x in hash_table[H] for y in hash_table[H] if x != y}
                        candidates_per_group[f"group_{current_group}"].update(pairs)
            # per band in second layer, check if the candidate pair is in every row!
            dict_candidate_pairs = defaultdict(int)
            for row_second_layer in range(self.r[1]):
                for candidate_pair in candidates_per_group[f"group_{row_second_layer}"]:
                    dict_candidate_pairs[candidate_pair] += 1
            candidate_pairs.update({candidate_pair for candidate_pair in dict_candidate_pairs if dict_candidate_pairs[candidate_pair] == self.r[1]})
        return candidate_pairs


    def plot_probability_function(self, step=0.001, plot_threshold=True, plot_fp=True, plot_fn=True):
        # define function
        _probability = lambda s : 1 - (1 - (1 - (1 - s**float(self.r[0]))**float(self.b[0]))**float(self.r[1]))**float(self.b[1])

        # create data
        x_array = np.arange(start=0, stop=1 + step, step=step)
        y_array = np.apply_along_axis(_probability, axis=0, arr=x_array)

        # create the figure and axis objects
        fig, ax = plt.subplots()
        ratio =  18.5 / 10.5
        x = 7
        fig.set_size_inches(x, x / ratio)

        # plot the data and customize
        ax.plot(x_array, y_array, label=r"$1 - (1-s^{" + str(self.r[0]) +"})^{" + str(self.b[0]) + "}$")
        ax.set_xlabel('s(x, y) (Jaccard similarity)')
        ax.set_ylabel('Pr[(x,y) is candidate pair]')


        if plot_threshold:
            plt.axvline(x=self.threshold, color='red', drawstyle='steps', linestyle='--')
            ax.set_title(f'S-curve (threshold = {self.threshold}, b={self.b[0]}, r={self.r[0]})')
        else:
            ax.set_title(f'S-curve (b={self.b[0]}, r={self.r[0]})')
        if plot_fp:
            x_smaller_than_threshold = list(x_array[x_array <= self.threshold])
            y_array_smaller_than_threshold = list(y_array[:len(x_smaller_than_threshold)])
            x_smaller_than_threshold.append(0.5)
            y_array_smaller_than_threshold.append(0)
            ax.fill(x_smaller_than_threshold, y_array_smaller_than_threshold, color=(239/255, 112/255, 198/255, 0.38), label='Pr(False Negative)')
        if plot_fn:
            x_greater_than_threshold = list(x_array[x_array > self.threshold])
            y_array_greater_than_threshold = list(y_array[-len(x_greater_than_threshold):])
            x_greater_than_threshold.append(0.5)
            y_array_greater_than_threshold.append(1)
            ax.fill(x_greater_than_threshold, y_array_greater_than_threshold,
                    color=(239 / 255, 147 / 255, 112 / 255, 0.38), label='Pr(False Positive)')
        ax.legend()
        plt.show()


class MinHashLSHInsertionSession:
    '''Context manager for batch insertion of documents into a MinHashLSH.
    '''

    def __init__(self, lsh, buffer_size):
        self.lsh = lsh
        self.lsh.buffer_size = buffer_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            hashtable.empty_buffer()

    def insert(self, key, minhash, check_duplication=True):
        '''
        Insert a unique key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by
        the key.

        Args:
            key (hashable): The unique identifier of the set.
            minhash (datasketch.MinHash): The MinHash of the set.
        '''
        self.lsh._insert(key, minhash, check_duplication=check_duplication,
                         buffer=True)


if __name__ == "__main__":
    from minhash import MinHash

    data = ["a", "b", "c"]
    data_1 = ["d", "b", "e", "c", "a"]

    m1 = MinHash(num_perm=1024)
    for el in data:
        m1.update(el.encode('utf-8'))
    m2 = MinHash(num_perm=1024)
    for el in data_1:
        m2.update(el.encode('utf-8'))
    print('created minhashes')
    lsh = MinHashLSHAmplified(threshold=0.58, num_perm=1024, amplified=True)
    lsh.insert(key='1', minhash=m1)
    lsh.insert(key='2', minhash=m2)
    print('getting candidate pairs')

    pairs_a = lsh.get_candidate_pairs()
    pairs_b = lsh.get_candidate_pairs_new()
