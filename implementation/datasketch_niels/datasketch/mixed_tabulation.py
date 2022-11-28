import numpy as np



# initialize 20-degree polynomial
# why default_rng? https://numpy.org/doc/stable/reference/random/index.html#random-quick-start

class PolyHash(object):

    def __init__(self, degree=20):
        self.degree = np.int32(20)
        self.mersenne_prime = np.int64(2305843009213693951)


        self.rng = np.random.default_rng(seed=1234)
        self.m_coef = np.repeat(np.iinfo(np.int64).max, repeats=self.degree)
        for i in range(0, self.degree):
            while self.m_coef[i] > self.mersenne_prime:
                self.m_coef[i]= self.rng.integers(low=0, high=np.iinfo(np.int64).max) >> 3

    def hash(self, x):
        hash = 0
        for i in range(self.degree - 1, -1, -1):
            print(i)

            hash = hash * x + self.m_coef[i]
            print(hash >> 61)
            hash = (hash & self.mersenne_prime) + (hash >> 61)
            print(hash)
        hash = (hash & self.mersenne_prime) + (hash >> 61)
        print(hash)
        return hash


class MixedTabulation(object):

    def __init__(self):
        self.mt_T1 = np.empty(shape=(256, 4), dtype=np.dtype(np.int64))
        self.mt_T2 = np.empty(shape=(256, 4), dtype=np.dtype(np.int32))
        self.polyhash = PolyHash(degree=20)
        x = 0
        for i in range(0, 4):
            for j in range(0, 256):
                x += 1
                self.mt_T1[j, i] = self.polyhash.hash(x)
                self.mt_T1[j, i] <<= 32
                x += 1
                self.mt_T1[j, i] += self.polyhash.hash(x)
                x += 1
                self.mt_T2[j, i] = self.polyhash.hash(x)


if __name__ == "__main__":
    ii64 = np.iinfo(np.int64).max
    ii64 = np.iinfo(np.int32).max

    ph = PolyHash()
    hash = ph.hash(32)
    type(hash)
    print(bin(753607659917372632))
    print(bin(2305843009213693951))
    753607659917372632 & 2305843009213693951

    print(2**32-1)

    mixed_tab = MixedTabulation()
