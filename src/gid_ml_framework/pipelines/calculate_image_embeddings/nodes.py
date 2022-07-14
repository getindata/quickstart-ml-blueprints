from typing import List, Union
from gid_ml_framework.image_embeddings.data.hm_data import HMDataset
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import mlflow
import torch
import numpy as np
import pandas as pd
from itertools import chain
# protobuf warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def _stack_predictions(predictions: List, emb_size: Union[int, str]) -> pd.DataFrame:

    out_emb, out_labels = list(), list()
    for emb, labels in predictions:
        out_emb.append(emb)
        out_labels.append(labels)
    article_ids = [article_id.split('.')[0] for article_id in chain(*out_labels)]
    column_names = [f'emb_{i+1}' for i in range(int(emb_size))]
    embeddings = torch.cat(out_emb).numpy()
    return pd.DataFrame(data=embeddings, index=article_ids, columns=column_names)

def calculate_image_embeddings(
    RUN_ID: str,
    img_dir: str,
    batch_size: int) -> None:

    client = MlflowClient()
    run = client.get_run(RUN_ID)
    print("run_id: {}".format(run.info.run_id))
    print("params: {}".format(run.data.params))
    print("status: {}".format(run.info.status))
    logged_model_uri = f'runs:/{RUN_ID}/model'
    loaded_model = mlflow.pytorch.load_model(logged_model_uri)

    hm_dataset = HMDataset(img_dir, transform=transforms.ToTensor())
    hm_dataloader = DataLoader(dataset=hm_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)

    trainer = pl.Trainer(max_epochs=1, logger=False)
    predictions = trainer.predict(loaded_model, dataloaders=hm_dataloader)

    df_embeddings = _stack_predictions(predictions, run.data.params['embedding_size'])
    return df_embeddings
