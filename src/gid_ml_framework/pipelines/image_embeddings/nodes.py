from gid_ml_framework.image_embeddings.data.hm_data import HMDataLoader
from gid_ml_framework.image_embeddings.model.pl_autoencoder_module import LitAutoEncoder
from gid_ml_framework.image_embeddings.model import pl_encoders, pl_decoders
import pytorch_lightning as pl
import mlflow.pytorch
from pytorch_lightning.utilities.seed import seed_everything


seed_everything(321, True)


def train_image_embeddings(
    img_dir: str,
    encoder: str,
    decoder: str,
    batch_size: int = 32,
    embedding_size: int = 32,
    num_epochs: int = 5,
    save_model: bool = False,
    model_name: str = "image_embeddings_model") -> None:

    hm_dataloader = HMDataLoader(img_dir, batch_size=batch_size)
    hm_encoder = getattr(pl_encoders, encoder)
    hm_decoder = getattr(pl_decoders, decoder)

    hm_autoencoder = LitAutoEncoder(
        encoder=hm_encoder(embedding_size),
        decoder=hm_decoder(embedding_size)
        )
    
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='train_loss',
        min_delta=0.00001,
        patience=5,
        verbose=True,
        mode='min'
    )

    hm_trainer = pl.Trainer(limit_train_batches=100,
        max_epochs=num_epochs, 
        logger=False, # disabling pytorch_lightning default logging, because we have mlflow
        enable_checkpointing=False,
        callbacks=[early_stop_callback])

    mlflow.pytorch.autolog(log_models=save_model, registered_model_name=model_name)

    hm_trainer.fit(model=hm_autoencoder, train_dataloaders=hm_dataloader)
