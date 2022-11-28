import pandas as pd
import numpy as np
from scipy.integrate import trapezoid as integrate
import json
tuples_of_interest = [(25, 64), (25,128), (25, 256), (25,512), (25,1024)]
list_df = []
for (iterations, no_perm) in tuples_of_interest:
    df_bootstrap = pd.read_excel(f"bootstrap_v2_{iterations}_n_{no_perm}.xlsx", index_col=0)

    # Average per hyperparameter combi
    if 'f1' in df_bootstrap.columns:
        df_bootstrap = df_bootstrap.drop(labels='f1', axis=1)
    df_average = df_bootstrap.groupby(["threshold", "amplified", "sketch_type", "amplified_x_sketch_type"], as_index=False).mean()
    df_average['pair_completeness'] = df_average['pair_completeness'].apply(lambda x: x/100)
    df_average['alpha'] = df_average['reduction_ratio'] * df_average['pair_completeness']
    df_average['num_perm'] = no_perm
    #dict_ = {'pair_completeness' : 1, 'reduction_ratio': 0, }
    #df_average.append(dict_)
    list_df.append(df_average)
df_results = pd.concat(list_df)



# group by num_perm, method, amplfiied
grouped = df_results.groupby(["num_perm", "amplified", "sketch_type"], as_index=False)

def extend_with_number(array, number):
    values = list(array.to_numpy())
    new_list = [number]
    new_list.extend(values)
    new_array = np.array(new_list)
    return new_array


grouped_auc = grouped.apply(lambda x: integrate(y=extend_with_number(x.pair_completeness,1), x=extend_with_number(x.reduction_ratio,0.5)))
grouped_auc.rename(columns={ grouped_auc.columns[3]: "auc" }, inplace = True)

grouped_2 = grouped_auc.groupby(["num_perm", "sketch_type"], as_index=False)
grouped_auc_perc_increase = grouped_2.apply(lambda x: (x.auc.max() - x.auc.min()) / x.auc.min() * 100)
grouped_auc_perc_increase.rename(columns={ grouped_auc_perc_increase.columns[2]: "auc_perc_increase" }, inplace = True)
grouped_auc_perc_increase['amplified'] = True

df_alpha_max = grouped["alpha"].agg(np.max)

df_joined = df_alpha_max.merge(df_results, how="inner", on=["num_perm", "amplified", "sketch_type", "alpha"])
df_joined = df_joined.merge(grouped_auc, how="left", on=["num_perm", "amplified", "sketch_type"])
df_joined = df_joined.merge(grouped_auc_perc_increase, how="left", on=["num_perm", "amplified", "sketch_type"])
df_joined["auc_perc_increase"] = df_joined["auc_perc_increase"].apply(lambda x: x if  pd.notna(x) else '-')
with open('./results/parameter_config.json', 'r') as file_:
    parameter_config_list = json.load(file_)
def add_parameters(row, parameter_config_list):
    params = next(x['params'] for x in parameter_config_list if (abs(x['threshold'] - row['threshold']) < 0.02
                                                            and x['num_perm'] == row['num_perm']
                                                            and (x['amplified'] is row['amplified'])))
    string_params = f'({params[1][0]},{params[1][1]},{params[0][0]},{params[0][1]})'
    return string_params
# add parameters

df_joined['params'] = df_joined.apply(lambda row: add_parameters(row, parameter_config_list), axis =1 )


df_latex = df_joined.loc[:,["num_perm", "amplified", "sketch_type", "alpha","threshold","pair_completeness", "reduction_ratio", "auc", "auc_perc_increase", "time_taken", 'params']]
df_latex["amplified"] = df_latex["amplified"].apply(lambda x: 'Yes' if x else 'No')
df_latex['sketch_type'] = df_latex["sketch_type"].apply(lambda x: 'FSS' if x == 'fss' else 'MinHash')


df_latex = df_latex.round(4)

print(df_latex.to_latex(index=False))