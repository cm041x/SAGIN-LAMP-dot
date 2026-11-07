from collections import OrderedDict

import torch


class NodePredictor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.readout = torch.nn.Sequential(
            OrderedDict(
                [
                    ("L1", torch.nn.Linear(input_dim, input_dim // 2)),
                    ("LeakyReLU1", torch.nn.LeakyReLU()),
                    ("L2", torch.nn.Linear(input_dim // 2, output_dim)),
                ]
            )
        )

        if input_dim < 2:
            self.readout = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("L1", torch.nn.Linear(input_dim, output_dim)),
                    ]
                )
            )

    def forward(self, x, **kwargs):
        return self.readout(x)


class LinkPredictor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.readout = torch.nn.Sequential(
            OrderedDict(
                [
                    ("L1", torch.nn.Linear(input_dim * 2, input_dim)),
                    ("LeakyReLU1", torch.nn.LeakyReLU()),
                    ("L2", torch.nn.Linear(input_dim, output_dim)),
                ]
            )
        )

    def forward(self, x: torch.Tensor, link_pred_ids, **kwargs):
        source, target = link_pred_ids
        x = torch.cat((x[source], x[target]), dim=-1)
        return self.readout(x)


def get_predictor(level, input_dim, output_dim):
    if level == "link":
        return LinkPredictor(input_dim, output_dim)
    return NodePredictor(input_dim, output_dim)
