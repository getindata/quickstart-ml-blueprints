import pandas as pd
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Tuple


log = logging.getLogger(__name__)

def _load_model(model_name: str) -> SentenceTransformer:
    """Loads model from `SentenceTransformers` framework.

    Args:
        transformer_model (str, optional): name of a model.

    Returns:
        SentenceTransformer: a model
    """
    log.info(f'Loading sentence transformer model: {model_name}')
    model = SentenceTransformer(model_name)
    return model

def prepare_desciptions_and_labels(articles: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Prepares data for embeddings generation.

    Args:
        articles (pd.DataFrame): article dataframe with `detail_desc` and `article_id`

    Returns:
        Tuple[List[str], List[str]]: article descriptions, article labels
    """
    article_descriptions = articles['detail_desc'].astype('str').to_list()
    article_labels = articles['article_id'].to_list()
    assert len(article_descriptions)==len(article_labels)
    log.info(f'There are {len(article_labels)} articles for embedding calculation')
    return article_descriptions, article_labels   

def generate_embeddings(descriptions: List[str], labels: List[str], transformer_model: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
    """Generates embeddings from descriptions using transformer model.

    Args:
        labels (List[str]): labels/id's for result dataframe index
        descriptions (List[str]): descriptions for embeddings generation
        transformer_model (str, optional): transformer model from SentenceTransformers framework. Defaults to 'all-MiniLM-L6-v2'.

    Returns:
        pd.DataFrame: text embeddings
    """
    log.info(f'First description: {descriptions[0]}, first label: {labels[0]}')
    model = _load_model(transformer_model)
    embeddings = model.encode(descriptions)
    log.info(f'Embeddings shape: {embeddings.shape}')
    text_embeddings = pd.DataFrame(data=embeddings, index=labels)
    text_embeddings.columns = [f'text_emb_{i+1}' for i, _ in enumerate(text_embeddings.columns)]
    return text_embeddings
