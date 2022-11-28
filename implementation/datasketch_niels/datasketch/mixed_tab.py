import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")
import pyMixedTabulation

class MixedTabulation(object):
    def __init__(self, seed=1):
        self.mixed_tab_object = pyMixedTabulation.PyMixTab(seed)

    def _create_64_bit_from_x_and_i(self, x, i):
        bin_x = bin(x)
        bin_i = bin(i)

        bin_x_part = bin_x.split('b')[1]
        zeros_to_add_x = 32 - len(bin_x_part)
        zeros_x = zeros_to_add_x * '0'
        bin_x_32_bit = zeros_x + bin_x_part

        bin_i_part = bin_i.split('b')[1]
        zeros_to_add_i = 32 - len(bin_i_part)
        zeros_i = zeros_to_add_i * '0'
        bin_i_32_bit = zeros_i + bin_i_part
        total_64_bit = '0b' + bin_x_32_bit + bin_i_32_bit

        total = int(total_64_bit, 2)
        return total

    def get_hash(self, x, i):
        """
        x should be smaller than max_32_bit integer
        i should be smaller than max_32_bit integer
        """
        key_64_bit = self._create_64_bit_from_x_and_i(x, i)
        hash = self.mixed_tab_object.getHash(key_64_bit)
        return hash


if __name__ == "__main__":
    mixtab = MixedTabulation(4)
    x = 5
    list_hashes_1 = []
    for i in range(1024):
        hash = mixtab.get_hash(x=7, i=i)
        list_hashes_1.append(hash)

    list_hashes_2 = []
    for i in range(1024):
        hash = mixtab.get_hash(x=7, i=i)
        list_hashes_2.append(hash)


    print(list_hashes_1 == list_hashes_2)

    import pandas as pd
    pd_hashes = pd.Series(list_hashes_1)

    pd_unique = pd_hashes.unique()

    even_list = []
    for el in list_hashes_1:
        if el % 2 == 0:
            even_list.append(el)

