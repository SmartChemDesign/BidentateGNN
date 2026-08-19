import torch
from torch import nn


class BasePooling(nn.Module):
    def __init__(self, input_dims):
        super(BasePooling, self).__init__()
        self.input_dims = input_dims
        self.output_dim = None

    def forward(self, *tensors):
        if len(tensors) != len(self.input_dims):
            raise ValueError(f"expected {len(self.input_dims)} tensors, but {len(tensors)} were given")


class ConcatPooling(BasePooling):
    def __init__(self, input_dims):
        super(ConcatPooling, self).__init__(input_dims)
        self.output_dim = sum(input_dims)

    def forward(self, *tensors):
        super().forward(*tensors)
        for i, tensor in enumerate(tensors):
            if tensor.shape[-1] != self.input_dims[i]:
                raise ValueError(
                    f"tensor {i} must have last dimension of {self.input_dims[i]}, but has {tensor.shape[-1]}")
        return torch.cat(tensors, dim=-1)


class SumPooling(BasePooling):
    def __init__(self, input_dims):
        super(SumPooling, self).__init__(input_dims)
        self.output_dim = sum(input_dims) // 2
        self.sequences = nn.ModuleList([
            nn.Linear(input_dim, self.output_dim)
            for input_dim in input_dims
        ])

    def forward(self, *tensors):
        super().forward(*tensors)
        tensors = [seq(tensor) for seq, tensor in zip(self.sequences, tensors)]
        return torch.sum(torch.cat([tensor.unsqueeze(0) for tensor in tensors], dim=0), dim=0)


class MaxPooling(BasePooling):
    def __init__(self, input_dims):
        super(MaxPooling, self).__init__(input_dims)
        self.output_dim = sum(input_dims) // 2
        self.sequences = nn.ModuleList([
            nn.Linear(input_dim, self.output_dim)
            for input_dim in input_dims
        ])

    def forward(self, *tensors):
        super().forward(*tensors)
        tensors = [seq(tensor) for seq, tensor in zip(self.sequences, tensors)]
        return torch.max(torch.cat([tensor.unsqueeze(0) for tensor in tensors], dim=0), dim=0)[0]


class CrossAttentionPooling(BasePooling):
    def __init__(self, input_dims):
        super(CrossAttentionPooling, self).__init__(input_dims)
        self.output_dim = sum(input_dims) // 2
        self.query_sequence = nn.Linear(input_dims[0], self.output_dim)
        self.key_value_sequence = nn.Linear(input_dims[1], self.output_dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.output_dim, num_heads=1, batch_first=True)

    def forward(self, tensor_0, tensor_1):
        super().forward(tensor_0, tensor_1)
        query = self.query_sequence(tensor_0)
        key = value = self.key_value_sequence(tensor_1)
        return self.attention(query, key, value, need_weights=False)[0]
