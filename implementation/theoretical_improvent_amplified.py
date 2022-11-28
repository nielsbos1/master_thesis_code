import json
from helper import add_error_info_to_param_list
with open('./results/parameter_config_extended.json', 'r') as file_:
    param_config_list = json.load(file_)

df_params = add_error_info_to_param_list(param_config_list=param_config_list)