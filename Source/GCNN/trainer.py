import numpy as np
import torch
from torch_geometric.data import Batch

from Source.trainer_logK import ModelTrainer



class GCNNTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(GCNNTrainer, self).__init__(*args, **kwargs)

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

        train, val = self.train_valid_data[0]
        train_batch = Batch.from_data_list(train.dataset + val.dataset)
        test_batch = Batch.from_data_list(self.test_data.dataset) if self.test_data else None

        train_pred = prepare(get_mean_pred(train_batch))
        train_true = {k: v.numpy() for k, v in train_batch.y.items()}

        test_pred = prepare(get_mean_pred(test_batch)) if self.test_data else None
        test_true = {k: v.numpy() for k, v in test_batch.y.items()} if self.test_data else None

        valid_batches = [Batch.from_data_list(self.train_valid_data[i][1].dataset) for i in range(len(self.models))]
        valid_preds = [model(batch) for model, batch in zip(self.models, valid_batches)]
        valid_pred = {target["name"]: torch.cat([pred[target["name"]] for pred in valid_preds], dim=0)
                      for target in self.targets}
        valid_pred = prepare(valid_pred)
        valid_true = {
            target["name"]: np.concatenate([batch.y[target["name"]].numpy() for batch in valid_batches])
            for target in self.targets
        }

        return (train_true, valid_true, test_true,
                train_pred, valid_pred, test_pred)
