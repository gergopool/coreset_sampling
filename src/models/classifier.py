import torch.nn as nn
import torchinfo


from src.models import backbones
from projection import projection

__all__ = ['classifier']

class Classifier(nn.Module):

    def __init__(self, str_backbone, num_classes=10):
        super(Classifier, self).__init__()
        self.backbone = backbones.__dict__[str_backbone]
        self.projector = projection(self.backbone.last_channel_num, 128)
        self.linear_classifier = nn.Linear(128, num_classes)

    def forward(self, inputs, return_z=False):

        # Extract features with backbone CNN network
        cnn_features = self.backbone(inputs)

        # Retrieve the representation and further projection
        z, projection = self.projector(cnn_features)

        # Predict final classes
        class_preds = self.linear_classifier(projection)

        if return_z:
            return class_preds, z

        return class_preds

def classifier(str_backbone, num_classes=10):
    return Classifier(str_backbone, num_classes=num_classes)


if __name__ == "__main__":

    from torchinfo import summary
    import sys

    model_name = sys.argv[1]
    model = classifier(model_name)
    summary(model, (1, 3, 32, 32))