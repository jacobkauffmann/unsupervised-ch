import torch
from torch.nn import functional as F
import torchmetrics
import pytorch_lightning as pl


class Model(pl.LightningModule):

    def __init__(self, model, epochs=200, learning_rate=0.001, weight_decay=1e-4, loss='bce', num_classes=10):
        super().__init__()
        assert loss in ['bce', 'ce']
        self.save_hyperparameters()
        self.model = model

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.valid_loss = torchmetrics.MeanMetric()

        self.loss = loss
        self.num_classes = num_classes
        self.epochs = epochs

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        if self.loss == 'ce':
            loss = F.cross_entropy(y_hat, y)
        elif self.loss == 'bce':
            y = F.one_hot(y, self.num_classes)
            loss = F.binary_cross_entropy_with_logits(y_hat, y.float())

        self.train_acc(y_hat, y)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/accuracy", self.train_acc, on_epoch=True, on_step=False)
        return {'loss': loss, 'accuracy': self.train_acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_hat = self.forward(x)
            if self.loss == 'ce':
                loss = F.cross_entropy(y_hat, y)
            elif self.loss == 'bce':
                y = F.one_hot(y, self.num_classes)
                loss = F.binary_cross_entropy_with_logits(y_hat, y.float())

        self.valid_loss(loss)
        self.valid_acc(y_hat, y)
        self.log("test/loss", self.valid_loss, on_epoch=True, on_step=False)
        self.log("test/accuracy", self.valid_acc, on_epoch=True, on_step=False)
        return {'loss': loss, 'accuracy': self.valid_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.hparams.learning_rate,
                                    momentum=0.9, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]
