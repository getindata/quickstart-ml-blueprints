import random

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytorch_lightning as pl
from torch import nn, optim


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, shuffle_reconstructions=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.shuffle_reconstructions = shuffle_reconstructions
        self.no_plot_images = 8

    def __reduce__(self):
        return (
            LitAutoEncoder,
            (self.encoder, self.decoder, self.shuffle_reconstructions),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat.flatten(), x.flatten())
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat.flatten(), x.flatten())
        self.log("val_loss", loss, on_epoch=True)
        return {"val_loss": loss, "x_hat": x_hat, "x": x}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        embedding = self.encoder(x)
        return embedding, y

    def validation_epoch_end(self, outputs):
        if self.shuffle_reconstructions:
            random_int = random.randint(0, len(outputs) - 1)
            results = outputs[random_int]
        else:
            results = outputs[0]
        self._save_reconstructed_plots(results, self.current_epoch, self.no_plot_images)

    @staticmethod
    def _save_reconstructed_plots(results, current_epoch, no_plot_images=8):
        fig, axarr = plt.subplots(no_plot_images, 2, figsize=(8, no_plot_images * 1.5))
        for i in range(no_plot_images):
            img_reconstructed = results["x_hat"][i, :].reshape(3, 128, 128)
            img_original = results["x"][i, :].reshape(3, 128, 128)
            axarr[i, 0].imshow(np.transpose(img_reconstructed.numpy(), (1, 2, 0)))
            axarr[i, 1].imshow(np.transpose(img_original.numpy(), (1, 2, 0)))
        mlflow.log_figure(
            plt.gcf(), f"reconstructed/img_reconstructed_{current_epoch}.png"
        )
        plt.close()
