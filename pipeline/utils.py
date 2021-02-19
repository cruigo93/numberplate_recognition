import yaml
import torch
import random
import os
import numpy as np
from collections import OrderedDict
import importlib
def load_pytorch_model(state_dict, *args, **kwargs):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('model.'):
            name = name.replace('model.', '') # remove `model.`
        new_state_dict[name] = v
    return new_state_dict

def load_yaml(file_name):
    with open(file_name, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_model_train_params(config):
    params = {}
    metrics = {}
    for k, v in config["EVALUATION_METRICS"].items():
        module = importlib.import_module(v["PY"])
        metric = getattr(module, v["CLASS"])(**v["ARGS"])
        metrics[k] = metric
    params["metrics"] = metrics
    
    module = importlib.import_module(config["AUGMENTATION"]["PY"])
    train_aug = getattr(module, config["AUGMENTATION"]["TRAIN"])()
    val_aug = getattr(module, config["AUGMENTATION"]["VAL"])()
    params["train_augs"] = train_aug
    params["val_augs"] = val_aug
    
    module = importlib.import_module(config["CRITERION"]["PY"])
    criterion = getattr(module, config["CRITERION"]["CLASS"])(**config["CRITERION"]["ARGS"])
    params["criterion"] = criterion
    
    module = importlib.import_module(config["MODEL"]["PY"])
    model = getattr(module, config["MODEL"]["ARCH"])(**config["MODEL"]["ARGS"])
    if len(config["RESUME"]) > 0:
        ckpt_path = config["RESUME"]
        ckpt_dict = torch.load(ckpt_path)   
        prev_state = utils.load_pytorch_model(ckpt_dict['state_dict'])
        model.load_state_dict(prev_state)
    params["model"] = model
    
    module = importlib.import_module(config["OPTIMIZER"]["PY"])
    optimizer = getattr(module, config["OPTIMIZER"]["CLASS"])(model.parameters(), **config["OPTIMIZER"]["ARGS"])
    params["optimizer"] = optimizer
    
    module = importlib.import_module(config["SCHEDULER"]["PY"])
    scheduler = getattr(module, config["SCHEDULER"]["CLASS"])(optimizer, **config["SCHEDULER"]["ARGS"])
    params["scheduler"] = scheduler
    
    module = importlib.import_module(config["MASK_POSTPROCESSING"]["PY"])
    postprocessing = getattr(module, config["MASK_POSTPROCESSING"]["CLASS"])(**config["MASK_POSTPROCESSING"]["ARGS"])
    params["postprocessing"] = postprocessing
    return params