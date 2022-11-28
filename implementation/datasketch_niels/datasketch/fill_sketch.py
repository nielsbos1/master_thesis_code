import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")
import pyMixedTabulation
import random
import time
from datasketch.hashfunc import sha1_hash32
import math
import numpy as np
from datasketch.minhash import MinHash
from datasketch.mixed_tab import MixedTabulation

_mersenne_prime = np.uint64((1 << 61) - 1)
_max_hash = np.uint64((1 << 32) - 1)
_hash_range = (1 << 32)


class FillSketch(object):
    def __init__(self, input, sketch_length=128, seeds=[1,2], hashfunc=sha1_hash32, mixedtab_objects=None, mixed_tab=True):
        if sketch_length > _hash_range:
            # Because 1) we don't want the size to be too large, and
            # 2) we are using 4 bytes to store the size value
            raise ValueError("Cannot have more than %d number of\
                    permutation functions" % _hash_range)
        self.list_seeds = None
        self.input = input
        self.mixed_tab = True if mixedtab_objects or mixed_tab else False
        # if self.mixed_tab:
        #     if mixedtab_objects:
        #         self.mixedtab_objects = mixedtab_objects
        #     else:
        #         self.mixedtab_objects = []
        #         for t in range(sketch_length):
        #             self.mixedtab_objects.append(pyhashniels.PyMixTab())
        if self.mixed_tab:
            if mixedtab_objects:
                self.mixedtab_object = mixedtab_objects
            else:
                self.mixedtab_object = [None, None]
                self.mixedtab_object[0] = MixedTabulation(seed=seeds[0])
                self.mixedtab_object[1] = MixedTabulation(seed=seeds[1])

        self.sketch_length = sketch_length
        # Check the hash function.
        if not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        self.hashfunc = hashfunc

        self.hash_outputs = {x: self.get_hash_values(x) for x in self.input}

        # generate hash
        self.hashvalues = self._generate_fill_sketch(input_set=input,
                                                     sketch_length=sketch_length)


    def _generate_fill_sketch(self, input_set, sketch_length, print_=True):
        sketch = np.repeat(math.inf, repeats=sketch_length)
        c = 0
        for i in range(2 * sketch_length):
            for input in input_set:
                if i < sketch_length:
                    bin_value = int(self.hash_outputs[input]["bins"][i])
                    v_value = i + self.hash_outputs[input]["values"][i]
                    # if print_:
                    #     print(f"b_{i}({input})={bin_value}, v_{i}({input}) = {v_value}")
                else:
                    bin_value = i - sketch_length
                    v_value = i + self.hash_outputs[input]["values"][i]
                    # if print_:
                    #     print(f"b_{i}({input})={bin_value}, v_{i}({input}) = {v_value}")
                if math.isinf(sketch[bin_value]):
                    c += 1
                sketch[bin_value] = min(sketch[bin_value], v_value)
                # if print_:
                #     print(sketch)
            if c == sketch_length:
                return sketch
        return sketch


    def get_hash_values(self, input):
        hash_value = self.hashfunc(input.encode('utf-8'))
        if self.mixed_tab:
            bins = np.empty(shape=self.sketch_length)
            for i in range(self.sketch_length):
                bins[i] = int(self.mixedtab_object[0].get_hash(x=hash_value, i=i) % self.sketch_length)
            values = np.empty(shape=self.sketch_length * 2)
            for i in range(self.sketch_length * 2):
                values[i] = self.mixedtab_object[1].get_hash(x=hash_value, i=i) / (2 ** 32 - 1)
            # print(bins)
            # print(values)
            # print(intermediate)
            # intermediate_copy = intermediate.astype('int64')
            # print(intermediate)
            # # bins = np.bitwise_and(intermediate, self.sketch_length - 1)
            # bins = intermediate % self.sketch_length
            # print(bins)
            # values = intermediate_copy / _mersenne_prime
            # print(values)
        else:
            a, b = self._init_permutations(self.sketch_length)
            intermediate = (a * hash_value + b) % _mersenne_prime
            bins = np.bitwise_and(intermediate, self.sketch_length - 1)
            values = intermediate / _mersenne_prime
            # phv = np.bitwise_and((a * hash_value + b) % _mersenne_prime, self.sketch_length - 1)
        return {"bins": bins, "values": values  }


    def _init_permutations(self, num_perm):
        # Create parameters for a random bijective permutation function
        # that maps a 32-bit hash value to another 32-bit hash value.
        # http://en.wikipedia.org/wiki/Universal_hashing
        gen = np.random.RandomState(seed=self.seed)
        return np.array([
            (gen.randint(1, _mersenne_prime, dtype=np.uint64), gen.randint(0, _mersenne_prime, dtype=np.uint64)) for _ in range(num_perm * 2)
        ], dtype=np.uint64).T

    def jaccard(self, other_fillsketch):
        return float(np.sum(self.hashvalues == other_fillsketch.hashvalues) / np.shape(self.hashvalues))


    def __len__(self):
        '''
        :returns: int -- The number of hash values.
        '''
        return len(self.hashvalues)

    def __eq__(self, other):
        '''
        :returns: bool -- If their seeds and hash values are both equal then two are equivalent.
        '''
        return type(self) is type(other) and \
            self.seed == other.seed and \
            np.array_equal(self.hashvalues, other.hashvalues)

    # @classmethod
    # def create_mixedtab_objects(cls, sketch_length, seed_gen):
    #     mixedtab_objects = []
    #     for t in range(sketch_length * 2):
    #         mixedtab_objects.append(pyhashniels.PyMixTab(seed_gen.get_single_seed()))
    #     return mixedtab_objects


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


if __name__ == "__main__":


    experiments = 1000
    sketch_length = 1 << 8
    storage = np.repeat(math.inf, experiments)
    storage_min = np.repeat(math.inf, experiments)
    j = 1
    random.seed(a=1)
    start = time.time()
    for j in range(experiments):
        if j % 10 == 0:
            print(j)
        iteration_seed_1 = random.randint(a=0, b=1 << 32 - 1)
        iteration_seed_2 = random.randint(a=0, b=1 << 32 - 1)
        input_set_a = {'1080p', '(45', '240hz', '1080', '3', '33.0', '46inch', '29.0', '.9inch', '240'}
        input_set_b = {'1080p', '240hz', '1080', '4', '33', '46inch', '29', '.9inch', '240'}
        # create mixedtab objects
        # mixedtab_objects = []
        # for t in range(sketch_length):
        #     mixedtab_objects.append(pyhashniels.PyMixTab())
        a = FillSketch(input=input_set_a, sketch_length=sketch_length, seeds=[iteration_seed_1, iteration_seed_2], mixed_tab=True)
        b = FillSketch(input=input_set_b, sketch_length=sketch_length, seeds=[iteration_seed_1, iteration_seed_2], mixed_tab=True)

        a_min = MinHash(num_perm=sketch_length, seed=iteration_seed_1)
        for el in input_set_a:
            a_min.update(el.encode('utf-8'))
        b_min = MinHash(num_perm=sketch_length, seed=iteration_seed_1)
        for el in input_set_b:
            b_min.update(el.encode('utf-8'))
        print(a.jaccard(b))
        storage[j] = a.jaccard(b)
        storage_min[j] = a_min.jaccard(b_min)
        # print(f'estimated jaccard (fill): {a.get_estimated_jaccard_similarity(b)}')
        # print(f'estimated jaccard (min): {a_min.jaccard(b_min)}')
        # print(f'actual jaccard: {jaccard(input_set_a, input_set_b)}')
    print(f'time: {time.time() - start}')
    print(f'mean: {np.mean(storage)}')
    print(f'mean: {np.mean(storage_min)}')

    import pandas as pd
    df_fill_sketch = pd.DataFrame(storage, columns=['jaccard'])
    df_min_hash = pd.DataFrame(storage_min, columns=['jaccard'])
    df_fill_sketch.hist()
    import seaborn as sns

    sns.histplot(data=df_fill_sketch, color='red', label='fill_sketch')
    sns.histplot(data=df_min_hash, color='green')

    import math
    real_jaccard = jaccard(input_set_a, input_set_b)
    df_fill_sketch['mean_substracted'] = df_fill_sketch['jaccard'] - real_jaccard
    df_fill_sketch['squared'] = df_fill_sketch['mean_substracted'].apply(lambda x: x ** 2)

    std_fill_sketch = math.sqrt(df_fill_sketch['squared'].mean())

    df_min_hash['mean_substracted'] = df_min_hash['jaccard'] - real_jaccard
    df_min_hash['squared'] = df_min_hash['mean_substracted'].apply(lambda x: x ** 2)

    std_min_hash = math.sqrt(df_min_hash['squared'].mean())
