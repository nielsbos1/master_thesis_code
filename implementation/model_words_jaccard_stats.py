import matplotlib.pyplot as plt
from seeds import seedsGen
from data_cleaning import data_cleaning_pipeline
import numpy as np
import os

seed_gen = seedsGen(file_loc='seeds.txt')
rng = np.random.default_rng(seed_gen.get_single_seed())
file_loc = './data/TVs-all-merged.json'

# seeds_bootstrap = seed_gen.get_batch_of_seeds(no_seeds=num_perm)
# extract model words
data_with_model_words, duplicates_pairs = data_cleaning_pipeline(file_loc)

list_len_model_words = [len(entity['model_words']) for uuid, entity in data_with_model_words.items()]

list_jaccard = []
for pair in duplicates_pairs:
    set_mw_1 = data_with_model_words[list(pair)[0]]['model_words']
    set_mw_2 = data_with_model_words[list(pair)[1]]['model_words']
    intersection = set_mw_1.intersection(set_mw_2)
    union = set_mw_1.union(set_mw_2)
    jaccard = len(intersection) / len(union)
    list_jaccard.append(jaccard)

average_model_words = sum(list_len_model_words) / len(list_len_model_words)
average_jaccard = sum(list_jaccard) / len(list_jaccard)
print(f'Average number of model words: {average_model_words}')
print(f'Average jaccard similarity duplicates: {average_jaccard}')

array_jaccard = np.array(list_jaccard)

list_random_jaccard = []
list_model_words = [entity['model_words'] for index, entity in data_with_model_words.items()]
for i in range(len(duplicates_pairs)):
    first_index = rng.integers(0, len(data_with_model_words))
    second_index = first_index
    while second_index == first_index:
        second_index = rng.integers(0, len(data_with_model_words))
    mw_1 = list_model_words[first_index]
    mw_2 = list_model_words[second_index]
    intersection = mw_1.intersection(mw_2)
    union = mw_1.union(mw_2)
    jaccard = len(intersection) / len(union)
    list_random_jaccard.append(jaccard)


array_random_jaccard = np.array(list_random_jaccard)

print(f'Average jaccard similarity random pairs: {array_random_jaccard.mean()}')


# Generate plot
plt.rcParams['text.usetex'] = True
plt.style.use('science')
figures_dir = r"C:\Users\Niels\Documents\thesis\latex\figures"

fig, ax = plt.subplots()
ratio = 18.5 / 10.5
x_width = 7
fig.set_size_inches(x_width, x_width / ratio)

ax.set_xlabel("Jaccard similarity")
ax.set_ylabel("Percentage of pairs that have a certain Jaccard similarity")
ax.set_xlim(0, 1)
ax.hist([array_jaccard, array_random_jaccard], bins=25, density=True,
        label=["True duplicates", "Random selection of pairs"])
ax.legend()

plt.savefig(os.path.join(figures_dir, f"jaccard_true_dup_random_pairs.png"), dpi=100)
