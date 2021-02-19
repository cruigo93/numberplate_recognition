import importlib
import numpy as np
from collections import defaultdict
import torch
from losses import dice_round

def dice_round_fn(predicted, ground_truth, score_threshold=0.5, area_threshold=0):
    """
    predicted, ground_truth: torch tensors
    """
    mask = predicted > score_threshold
    #     mask[mask.sum(dim=(1,2,3)) < area_threshold, :,:,:] = torch.zeros_like(mask[0])
    if mask.sum() < area_threshold:
        mask = torch.zeros_like(mask)
    #     print(1 - dice_round(mask, ground_truth).item())
    return 1 - dice_round(mask, ground_truth).item()

class Evaluator:
    def __init__(self, config):
        
        # self.metric = dice_round_fn
        self.config = config

        module = importlib.import_module(config["CRITERION"]["PY"])
        criterion = getattr(module, config["CRITERION"]["CLASS"])(**config["CRITERION"]["ARGS"])
        self.criterion = criterion

        self.th = config["EVALUATOR"]["THRESHOLD"]
        self.area = config["EVALUATOR"]["AREA"]

        self.area_list = config["EVALUATOR"]["AREA_SEARCH_LIST"]
        self.threshold_list = config["EVALUATOR"]["THRESHOLD_SEARCH_LIST"]

    def loss_fn(self, gt, pr):
        # print(gt.shape, pr.shape)
        return self.criterion(pr, gt)

    def metric_fn(self, outputs, labels, combo=False):
        metric = dice_round_fn(outputs, labels, self.th, self.area)
        return metric

    def batch_metric(self, eval_list):
        for outputs, labels in eval_list:
            metric = self.metric_fn(outputs, labels)
        return metric

    def set_params(self, best_area, best_threshold):
        self.th = best_threshold
        self.area = best_area

    def area_threshold_search(self, eval_list):
        best_score = 0
        best_threshold = 0
        best_area = 0
        for area in self.area_list:
            for threshold in self.threshold_list:
                score_list = []
                for outputs, labels in eval_list:
                    score = dice_round_fn(outputs, labels, score_threshold=threshold, area_threshold=area)
                    score_list.append(score)
                final_score = torch.mean(torch.FloatTensor(score_list))
                if final_score > best_score:
                    best_score = final_score
                    best_area = area
                    best_threshold = threshold
        return best_area, best_threshold, best_score

