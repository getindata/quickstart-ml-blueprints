import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from kedro.io import AbstractDataSet
from kedro.io.core import get_protocol_and_path
from pathy import Pathy

from recommender_gnn.extras.graph_utils.dgsr_utils import (
    SubGraphsDataset,
    create_graphs_list,
    load_graphs_python,
    save_graphs_python,
)

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
            dir: The directory location to load/save data.
            save_args: Additional arguments to kedro AbstractDataSet saving function
            load_args: Additional arguments to kedro AbstractDataSet loading function
        """
        self._dir = _create_path_obj(dir)
        _create_parent_dirs(self._dir)
        protocol, _ = get_protocol_and_path(dir)
        self._protocol = protocol

        self._load_args = dict()
        if load_args is not None:
            self._load_args.update(load_args)

        self._save_args = dict()
        if save_args is not None:
            self._save_args.update(save_args)

        self.file_extension = self._save_args.get("file_extension")

    def _load(self) -> SubGraphsDataset:
        load_path = self._dir
        dataset = SubGraphsDataset(load_path, load_graphs_python, self.file_extension)
        return dataset

    def _save(self, data: List) -> None:
        graphs_collection = self._unpack_data(data)
        save_filepath = os.path.join(self._dir, f"graphs.{self.file_extension}")
        logger.info(f"Saving graphs here: {self._dir}")
        save_graphs_python(save_filepath, graphs_collection)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._dir, protocol=self._protocol)

    @staticmethod
    def _unpack_data(data: List) -> Dict[str, Any]:
        """Unpacks the data from the list of tuples into a dictionary."""
        graphs_collection = {}
        if data:
            for row in data:
                if row:
                    user, item_number, graph, graph_dict = row
                    graph_id = "_".join([str(user), str(item_number)])
                    graphs_list = create_graphs_list(graph, graph_dict)
                    graphs_collection[graph_id] = graphs_list
        return graphs_collection


def _create_path_obj(path: str) -> Union[Pathy, Path]:
    path_obj = Pathy(path) if path[0:5] == "gs://" else Path(path)
    return path_obj


def _create_parent_dirs(path: str) -> None:
    if path.exists():
        pass
    else:
        logger.info(f"Creating new directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
