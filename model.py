import torch
import torch.nn as nn


class VideoClassificationModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_classes = cfg['num_classes']
        self.path_torchvideo_model = cfg['path_torchvideo_model']
        self.name_torchvideo_model = cfg['name_torchvideo_model']
        self.n_nodes = cfg["n_nodes"]
        self.freeze_layers = cfg.get("freeze_layers", 1.0) > 0.0
        self.dropout = cfg["dropout"]
        self.base_model = torch.hub.load(self.path_torchvideo_model, self.name_torchvideo_model, pretrained=True)

        if self.freeze_layers:
            print("Freeze layers of pretrained model")
            for name, param in self.base_model.named_parameters():
                param.requires_grad = False

        #change the final part of the base model with the desidered number of classes at the output
        if self.name_torchvideo_model == "slow_r50":
            self.base_model.blocks[5].proj = nn.Sequential(nn.Linear(2048, self.n_nodes),
                                                           nn.ReLU(),
                                                           nn.Dropout(self.dropout),
                                                           nn.Linear(self.n_nodes, self.num_classes))
        elif self.name_torchvideo_model == "slowfast_r50":
            self.base_model.blocks[6].proj = nn.Sequential(nn.Linear(2304, self.n_nodes),
                                                           nn.ReLU(),
                                                           nn.Dropout(self.dropout),
                                                           nn.Linear(self.n_nodes, self.num_classes))

        for name, param in self.base_model.named_parameters():
            print("Layer name: {} - requires_grad: {}".format(name, param.requires_grad))

    # forward function
    def forward(self, x):
        x = self.base_model(x)
        return x