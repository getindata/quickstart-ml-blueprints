import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dgl import load_graphs, save_graphs
from kedro.io import AbstractDataSet
from kedro.io.core import get_protocol_and_path

from gid_ml_framework.extras.graph_processing.dgsr import SubGraphsDataset

logger = logging.getLogger(__name__)


class DGSRSubGraphsDataSet(AbstractDataSet):
    """``DGSRSubGraphsDataSet`` loads/saves dgl graphs structures from/to a given directory using dgl loaders.

    Examples:
    ::
        >>> DGSRSubGraphsDataSet(dir='path/to/graphs/')
    """

    def __init__(
        self,
        dir: str,
        save_args: Dict[str, Any] = None,
        load_args: Dict[str, Any] = None,
    ):
        """Creates a new instance of DGSRSubGraphsDataSet to loads/saves dgl graphs structures from/to a given directory
        using dgl loaders.

        Args:
            dir: The directory with images to load/save data.
        """
        self._dir = Path(dir)
        protocol, _ = get_protocol_and_path(dir)
        self._protocol = protocol

        self._load_args = dict()
        if load_args is not None:
            self._load_args.update(load_args)

        self._save_args = dict()
        if save_args is not None:
            self._save_args.update(save_args)

    def _load(self) -> SubGraphsDataset:
        load_path = self._dir
        dataset = SubGraphsDataset(load_path, load_graphs)
        return dataset

    def _save(self, data: List) -> None:
        file_extension = self._save_args.get("file_extension")

        save_path = self._dir
        if save_path.exists():
            logger.warning("Directory already exists, it may be not empty!")
        else:
            logger.info(f"Creating new directory: {save_path}")
            save_path.mkdir(parents=False, exist_ok=False)

        for row in data:
            if row:
                user, item_number, graph, graph_dict = row
                save_dir = os.path.join(save_path, str(user))
                file_name = "_".join([str(user), str(item_number)])
                save_filepath = os.path.join(save_dir, f"{file_name}.{file_extension}")
                save_graphs(save_filepath, graph, graph_dict)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._dir, protocol=self._protocol)
