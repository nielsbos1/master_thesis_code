from seeds import seedsGen
from data_cleaning import data_cleaning_pipeline
from tqdm import tqdm
from datasketch_niels import datasketch
from collections import defaultdict
import numpy as np
import pandas as pd


seed_gen = seedsGen(file_loc='seeds.txt')
file_loc = './data/TVs-all-merged.json'

# seeds_bootstrap = seed_gen.get_batch_of_seeds(no_seeds=num_perm)
# extract model words
data_with_model_words, duplicates_pairs = data_cleaning_pipeline(file_loc)

# compute actual jaccard similarity
list_jaccard = []
for pair in duplicates_pairs:
    el_1 = data_with_model_words[list(pair)[0]]
    el_2 = data_with_model_words[list(pair)[1]]

    intersection = el_1['model_words'].intersection(el_2['model_words'])
    union = el_1['model_words'].union(el_2['model_words'])
    jaccard = len(intersection) / len(union)
    list_jaccard.append(jaccard)

jaccard_scores = pd.Series(list_jaccard)

jaccard_scores.hist()

# num_perm_list = [16, 32, 64, 128, 256, 512, 1024]
num_perm_list = [16, 32, 64, 128, 256, 512, 1024]

bootstrap_iter = 10
list_results = []
for num_perm in tqdm(iterable=num_perm_list, desc="length of sketch process"):
    dict_results_fss = defaultdict(list)
    dict_results_minhash = defaultdict(list)
    for iter_num in tqdm(iterable=range(bootstrap_iter), desc="bootstrap process"):
        for pair in duplicates_pairs:
            set_mw_1 = data_with_model_words[list(pair)[0]]['model_words']
            set_mw_2 = data_with_model_words[list(pair)[1]]['model_words']

            intersection = set_mw_1.intersection(set_mw_2)
            union = set_mw_1.union(set_mw_2)
            jaccard = len(intersection) / len(union)

            seed_minhash = seed_gen.get_single_seed()
            seed_mixed_tab_1 = seed_gen.get_single_seed()
            seed_mixed_tab_2 = seed_gen.get_single_seed()
            seeds_mixedtab = [seed_mixed_tab_1, seed_mixed_tab_2]

            minhash_1 = datasketch.MinHash(num_perm=num_perm, seed=seed_minhash)
            for model_word in set_mw_1:
                minhash_1.update(model_word.encode('utf-8'))

            minhash_2 = datasketch.MinHash(num_perm=num_perm, seed=seed_minhash)
            for model_word in set_mw_2:
                minhash_2.update(model_word.encode('utf-8'))

            fill_sketch_1 = datasketch.FillSketch(input=set_mw_1, seeds=seeds_mixedtab, sketch_length=num_perm, mixed_tab=True)
            fill_sketch_2 = datasketch.FillSketch(input=set_mw_2, seeds=seeds_mixedtab, sketch_length=num_perm, mixed_tab=True)

            jaccard_minhash = minhash_1.jaccard(minhash_2)

            jaccard_fss = fill_sketch_1.jaccard(fill_sketch_2)

            dict_results_fss[frozenset(pair)].append(jaccard_fss - jaccard)
            dict_results_minhash[frozenset(pair)].append(jaccard_minhash - jaccard)
# compute average deviations
    def compute_statistics(dict_results):
        absolute_deviations = {key: [abs(x) for x in value]  for key, value in dict_results.items()}
        average_deviation_over_bootstrap = [sum(value) / len(value)  for key, value in absolute_deviations.items()]
        average_absolute_deviation = np.mean(np.absolute(average_deviation_over_bootstrap))
        average_variance = np.mean(np.square(average_deviation_over_bootstrap))
        standard_deviation = np.sqrt(average_variance)
        return average_absolute_deviation, average_variance, standard_deviation

    minhash_abs_dev, minhash_var, minhash_std = compute_statistics(dict_results_minhash)
    fss_abs_dev, fss_var, fss_std = compute_statistics(dict_results_fss)

    minhash_results = {"sketch_length": num_perm, "sketch": "minhash", "absolute_deviation": minhash_abs_dev, "variance": minhash_var, "standard_deviation": minhash_std}
    fss_results = {"sketch_length": num_perm, "sketch": "fss", "absolute_deviation": fss_abs_dev, "variance": fss_var, "standard_deviation": fss_std}

    list_results.append(minhash_results)
    list_results.append(fss_results)


df_results = pd.DataFrame(list_results)

df_results.to_excel('minhash_vs_fss_preciseness.xlsx')