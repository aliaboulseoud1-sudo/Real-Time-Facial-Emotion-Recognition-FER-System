import torch
import torch.nn as nn
from torchvision import models


class ModelBuilder(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        model_name: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_p: float = 0.5,
        print_summary: bool = True
    ):
        super(ModelBuilder, self).__init__()

        self.model_name = model_name.lower()

        if self.model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)

        elif self.model_name == "mobilenetv2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.model = models.mobilenet_v2(weights=weights)

        else:
            raise ValueError(f"Model '{model_name}' not supported. Choose 'resnet18' or 'mobilenetv2'.")

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        if self.model_name == "resnet18":
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(in_features, num_classes)
            )

        elif self.model_name == "mobilenetv2":
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(in_features, num_classes)
            )

        self._init_weights()

        if print_summary:
            self._print_model_info(freeze_backbone)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _print_model_info(self, freeze_backbone: bool):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())

        print("\n================ MODEL SUMMARY ================")
        print(f"Model Name         : {self.model_name}")
        print(f"Pretrained         : {'Yes' if any(p.requires_grad for p in self.parameters()) else 'No'}")
        print(f"Backbone Frozen    : {'Yes' if freeze_backbone else 'No'}")
        print(f"Total Parameters   : {total_params:,}")
        print(f"Trainable Params   : {trainable_params:,}")
        print("================================================\n")

    def forward(self, x):
        return self.model(x)