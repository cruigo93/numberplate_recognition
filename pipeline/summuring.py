import pandas as pd
import numpy as np
import json
import random
import torch
import torchvision
import cv2


class SummuryWrite:
    def __init__(self, summury_filename, test_filename):
        self.summury_filename = summury_filename
        self.test_filename = test_filename

    def write_epoch(self, history: dict, epoch: int) -> None:
        # print(history)
        new_history = {}
        for k, v in history.items():
            new_history[k] = [v]
        history = new_history
        epoch_df = pd.DataFrame(data=history)
        if epoch < 2:
            # print("HHHH")
            epoch_df.to_csv(self.summury_filename, index=None)
        else:
            df = pd.read_csv(self.summury_filename)
            # print(df.head())
            df = df.append(epoch_df)
            # print(df.head())
            df.to_csv(self.summury_filename, index=None)

    def write_test_results(self, test_history: dict) -> None:
        new_history = {}
        for k, v in test_history.items():
            new_history[k] = [v]
        df = pd.DataFrame(data=new_history)
        df.to_csv(self.test_filename, index=None)
    
    def make_img_from_tensor(self, img_name, mask):
        pass

    def make_grid(self, best_area, best_threshold, eval_list, k=10, names=None):
        true_grid = []
        pred_grid = []

        colors = [
            [255,0,0],  
            [0,255,0],  
            [0,0,255],  
            [255,255,0],
            [0,255,255],
            [102,51,0]  
        ]

        if not names:
            sample_eval_list = random.sample(eval_list, k=k)
            
            for outputs, labels in sample_eval_list:
                probas = torch.sigmoid(outputs)
                masks = probas > best_threshold
                if masks.sum() < best_area:
                    masks = torch.zeros_like(masks)
                # new_labels = []
                # new_masks = []
                # for i in range(labels.shape[0]):
                #     label = labels[i]
                #     mask = masks[i]
                #     new_mask = np.zeros_like(mask)
                #     new_label = np.zeros_like(label)
                #     for j in range(label.shape[0]):
                #         new_mask[mask == j] =
                # print(torch.unique(masks))
                masks = masks.type(torch.uint8)
                true_grid.append(labels*255)
                pred_grid.append(masks*255)
                # print(torch.unique(masks))
            
            pred_grid = torch.cat(pred_grid, dim=0)
            true_grid = torch.cat(true_grid, dim=0)
            # print(torch.unique(pred_grid), pred_grid.shape)
            # print(torch.unique(true_grid), true_grid.shape)

            return {
                "pred": torchvision.utils.make_grid(pred_grid),
                "true": torchvision.utils.make_grid(true_grid)
            }
        else:
            sample = random.sample(np.arange(len(eval_list)).tolist(), k=k)
            for idx in sample:
                sample_eval_list = eval_list[idx]
                sample_names = names[idx]
                outputs, labels = sample_eval_list
                img_name, label_name = sample_names
                probas = torch.sigmoid(outputs)
                masks = probas > best_threshold
                if masks.sum() < best_area:
                    masks = torch.zeros_like(masks)
                mask = masks[0,:,:,:].cpu().detach().squeeze().numpy()
                mask = mask.astype('uint8')
                label = labels[0,:,:,:].cpu().detach().squeeze().numpy()
                label = label.astype(np.uint8)
                # print(img_name[0])
                # print(label_name[0])
                img = cv2.imread(img_name[0], cv2.IMREAD_UNCHANGED)
                img = ((img / 80) * 255).astype(np.uint8)
                # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                mask = np.stack((mask, mask, mask), axis=2) * np.array(colors[0])
                mask = mask.astype(np.uint8)
                img_pred = cv2.addWeighted(mask,0.7, img, 0.3, 0.)

                label = np.stack((label, label, label), axis=2) * np.array(colors[0])
                label = label.astype(np.uint8)
                img_label = cv2.addWeighted(label,0.7, img, 0.3, 0.)

                true_grid.append(img_label)
                pred_grid.append(img_pred)
            return {
                "pred": np.hstack(pred_grid),
                "true": np.hstack(true_grid)
            }
