import argparse
import sys
from functools import partial
from pytorch_lightning import Trainer

from src.data.samplers.k_uniform import KUniform
from src.models import get_trainer
from src.data import get_cifar10_dataloaders
from src.data.samplers import KUniform

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('model', help='Model')
    parser.add_argument('--seed', help='Seed to sampler')
    parser.add_argument('--k', help='K sample from each class', default=1000)
    parser.add_argument('--save-path', help='Path to save model and csv', default='log')
    parser.add_argument('--img-root', help='Path to pytorch downloaded images (e.g. cifar10)', default='/data/pytorch')
    parser.add_argument('--epochs', help="Number of epochs", default=200)
    parser.add_argument('--batch-size', help="Number of epochs", default=128)
    return parser.parse_args(args)



if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])

    model = get_trainer(args.model)
    sampler = partial(KUniform, k=args.k)
    train_loader, test_loader = get_cifar10_dataloaders(args.img_root,
                                                        args.batch_size,
                                                        sampler,
                                                        num_workers=6)

    trainer = Trainer(progress_bar_refresh_rate=20, max_epochs=args.epochs, gpus=1)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=test_loader)