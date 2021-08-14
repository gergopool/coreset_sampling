import torch
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from src.models.classifier import get_classifier

__all__ = ['get_trainer']

class LightningTrainer(LightningModule):

    def __init__(self, classifier):
        super(LightningTrainer, self).__init__()
        self.classifier = classifier
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, return_z=False):
        return self.classifier(x, return_z=return_z)

    def training_step(self, batch, _):
        x, y = batch
        pred_y = self(x)
        loss = self.cross_entropy_loss(pred_y, y)
        acc = accuracy(pred_y, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        pred_y = self(x)
        loss = self.cross_entropy_loss(pred_y, y)
        acc = accuracy(pred_y, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, _):
        self.evaluate(batch, 'val')

    def test_step(self, batch, _):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.1,
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-4,
        )

        scheduler_dict = {
            'scheduler': MultiStepLR(optimizer,
                                     milestones=[50,100,150],
                                     gamma=0.1),
            'interval': 'epoch',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

def get_trainer(str_backbone, num_classes=10):
    classifier = get_classifier(str_backbone, num_classes=num_classes)
    return LightningTrainer(classifier)


if __name__ == "__main__":

    from torchinfo import summary
    import sys

    model_name = sys.argv[1]
    model = get_trainer(model_name)
    summary(model, (1, 3, 32, 32))