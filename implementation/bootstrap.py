import json
import logging
import sys
import numpy as np
import pandas as pd
from seeds import seedsGen
from data_cleaning import data_cleaning_pipeline
from helper import  add_sketches, get_pair_quality, apply_lsh_generate_candidate_pairs
from helper import filter_candidate_pairs_brands_and_shops, get_duplicates_found_pairwise, get_total_possible_comparisions
from tqdm import tqdm
from time import perf_counter

_max_hash = np.uint64((1 << 32) - 1)

log_format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(stream=sys.stdout,
                    filemode="w",
                    format=log_format,
                    level=logging.INFO)


def bootstrap(num_perm, iterations, sample_ratio=2/3):
    seed_gen = seedsGen(file_loc='seeds.txt')
    file_loc = './data/TVs-all-merged.json'

    seeds_bootstrap = seed_gen.get_batch_of_seeds(no_seeds=num_perm)
    # extract model words
    data_with_model_words, duplicates_pairs = data_cleaning_pipeline(file_loc)
    list_df = []
    for iter_num in tqdm(iterable=range(iterations),desc="Bootstrap process"):

        # set-up generator with seed
        rng = np.random.default_rng(seed=seeds_bootstrap[iter_num])

        # sample data
        indices = rng.choice(a=len(data_with_model_words), size=int(sample_ratio * len(data_with_model_words)), replace=False)
        uuids = [uuid for uuid, value in data_with_model_words.items()]
        sampled_uuids = [uuid for index, uuid in enumerate(uuids) if index in list(indices)]
        data_sampled = {key: value for key, value in data_with_model_words.items() if key in sampled_uuids}

        # filter duplicates set
        duplicates_sampled = [x for x in duplicates_pairs if (list(x)[0] in sampled_uuids) and (list(x)[1] in sampled_uuids)]

        # add sketches
        data_with_sketches = add_sketches(data_sampled, seed_gen=seed_gen, num_perm=num_perm, mixed_tab=True)

        # retrieve parameters
        with open('./results/parameter_config.json', 'r') as file_:
            parameter_config_list = json.load(file_)

        # BOOTSTRAP!
        thresholds = list(list(np.arange(start=0.05, stop=1, step=0.05)))
        list_metrics = []
        for threshold in tqdm(iterable=thresholds, desc="Threshold process"):
            for amplified in [False, True]:
                for sketch_type in ['fss', 'minhash']:
                    start_time = perf_counter()
                    candidate_pairs = apply_lsh_generate_candidate_pairs(data_with_sketches=data_with_sketches,
                                                                         threshold=threshold, num_perm=num_perm,
                                                                         amplified=amplified, sketch_type=sketch_type,
                                                                         parameter_config_list=parameter_config_list)
                    candidate_pairs_filtered = filter_candidate_pairs_brands_and_shops(candidate_pairs=candidate_pairs,
                                                                                       data_with_sketches=data_with_sketches)
                    dict_ = {}
                    duplicates_found = get_duplicates_found_pairwise(candidate_pairs, duplicates_sampled)
                    pair_quality = get_pair_quality(candidate_pairs=candidate_pairs_filtered, data=data_with_sketches)
                    pair_completeness = len(duplicates_found) / len(duplicates_sampled) * 100
                    dict_['threshold'] = threshold
                    dict_['pair_quality'] = pair_quality
                    dict_['pair_completeness'] = pair_completeness
                    dict_['amplified'] = amplified
                    dict_['no_candidate_pairs'] = len(candidate_pairs_filtered)
                    dict_['sketch_type'] = sketch_type
                    dict_['iter_num'] = iter_num
                    dict_['time_taken'] = perf_counter() - start_time
                    list_metrics.append(dict_)
        no_possible_comparisons = get_total_possible_comparisions(data_sampled)
        df_results = pd.DataFrame(list_metrics)
        df_results['reduction_ratio'] = df_results['no_candidate_pairs'].apply(lambda x: (no_possible_comparisons - x) / no_possible_comparisons)
        # df_results['f1'] = df_results.apply(lambda x: (2 * x['pair_quality'] * x['pair_completeness'] / 10000)/(x['pair_quality'] / 100 + x['pair_completeness']/100), axis =1)
        df_results['amplified_x_sketch_type'] = df_results.apply(lambda x: ('amplified' if x['amplified'] else 'not amplified') + ' - ' + x['sketch_type'], axis =1)
        list_df.append(df_results)
    bootstrap_results = pd.concat(list_df)
    bootstrap_results.to_excel(f"bootstrap_v2_{iterations}_n_{num_perm}.xlsx")
    return list_df




if __name__ == "__main__":
    num_perm = 64
    iterations = 25
    list_df = bootstrap(num_perm=num_perm, iterations=iterations)

