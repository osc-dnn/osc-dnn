import configparser
import os

__args = dict(
    hidden_units='[[0,0],[0,0]]',
    cv='5',
    stratified = 'T',
    learning_rate = '0',
    dropout = '0',
    normalization = 'T',
    pca = 'T',
    pcaVec = '0',
    batch_size = '0',
    train_steps = '0',
    reorg_norm_factor = '0',
    train_percent = '1'
)

__files = dict(
    csv_output_file = 'output_filename',
    csv_features_file = 'csvfile_with_column_and_index_labels',
    csv_y_file = 'csvfile_with_target_values',
)

args_keys = __args.keys()
files_keys = __files.keys()
path = "settings.ini"

def create_config(path):
    config = configparser.ConfigParser()
    config.add_section("Settings")
    for key in __args:
        config.set("Settings", key, __args[key])
    for key in __files:
        config.set("Settings", key, __files[key])

    with open(path, "w") as config_file:
        config.write(config_file)
  
def get_config(path):
    if not os.path.exists(path):
        create_config(path)
 
    config = configparser.ConfigParser()
    config.read(path)
    return config 
 
def get_setting(path, section, setting):
    config = get_config(path)
    value = config.get(section, setting)
    return value
  
def update_setting(path, section, setting, value):
    config = get_config(path)
    config.set(section, setting, str(value))
    with open(path, "w") as config_file:
        config.write(config_file)

