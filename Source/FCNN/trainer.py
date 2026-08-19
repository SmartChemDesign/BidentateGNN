import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from Source.trainer import ModelTrainer


class FCNNTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(FCNNTrainer, self).__init__(*args, **kwargs)

    def get_true_pred(self):
        def get_mean_pred(batch):
            all_pred = [model(batch) for model in self.models]
            output = {
                target["name"]: torch.cat([
                    pred[target["name"]].unsqueeze(-1) for pred in all_pred
                ], dim=-1).mean(dim=-1)
                for target in self.targets
            }
            return output

        def prepare(pred):
            for target in self.targets:
                key = target["name"]
                if target["mode"] == "binary_classification":
                    pred[key] = (pred[key] > 0.5).to(torch.int32)
                elif target["mode"] == "multiclass_classification":
                    pred[key] = torch.argmax(pred[key], dim=-1)
                pred[key] = pred[key].detach().numpy()
            return pred

        def get_mean_predictions(dataset):
            true = {}
            pred = {}
            for x, target in dataset:
                pred_target = get_mean_pred(x)
                for target_name, target_value in target.items():
                    if target_name not in true:
                        true[target_name] = []
                    true[target_name] += [target_value]
                for target_name, target_value in pred_target.items():
                    if target_name not in pred:
                        pred[target_name] = []
                    pred[target_name] += [target_value]

            for target_name, target_value in true.items():
                true[target_name] = torch.cat(target_value, dim=0).numpy()

            for target_name, target_value in pred.items():
                pred[target_name] = torch.cat(target_value, dim=0)

            return true, prepare(pred)

        train, val = self.train_valid_data[0]
        train_true, train_pred = get_mean_predictions(train.dataset + val.dataset)
        test_true, test_pred = get_mean_predictions(self.test_data.dataset)

        valid_preds = []
        valid_trues= []
        for (train, val), model in zip(self.train_valid_data, self.models):
            valid_preds += [model(x) for x, target in val.dataset]
            valid_trues += [target for x, target in val.dataset]

        valid_pred = prepare({
            target["name"]: torch.cat([
                pred[target["name"]] for pred in valid_preds
            ], dim=0)
            for target in self.targets
        })

        valid_true = {
            target["name"]: torch.cat([
                pred[target["name"]] for pred in valid_trues
            ], dim=0).numpy()
            for target in self.targets
        }


        return (train_true, valid_true, test_true,
                train_pred, valid_pred, test_pred)
