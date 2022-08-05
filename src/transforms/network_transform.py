"""Transforms for loading and applying a neural network to input data."""
from typing import Any, Dict, Hashable, List
import monai.transforms.transform
import torch


class NetworkForwardTransform(monai.transforms.transform.Transform):
    # pylint: disable=too-few-public-methods
    """Load a network and apply it to (some of) the input.

    :param network_class: Class of the network that will be created.
    :param network_arguments: Keyword rguments to pass to the network during construction.
    :param network_path: Path to a state dict file to load the network parameters from.
    :param input_keys: Keys of the data dictionary that will be used as input to the network.
        Can be 'None' for positional inputs that need to be skipped.
    :param output_key: Key that the output of the network will be written to.
    :param make_batch: Whether to convert the input data to batch format.
    :param device: Device to transfer the network and data to.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        network_path: str = None,
        input_keys: List = None,
        output_key: str = "net_out",
        make_batch=True,
        device: str = "cuda",
    ):  # pylint: disable=too-many-arguments
        super().__init__()
        self.network: torch.nn.Module = network
        if network_path is not None:
            self.network.load_state_dict(torch.load(network_path))
        self.network.eval()
        self.device = device
        self.network.to(self.device)
        self.input_keys = input_keys if input_keys is not None else [None, "image"]
        self.output_key = output_key
        self.make_batch = make_batch

    def __call__(self, d: Dict[Hashable, Any]):
        input_list = []
        for key in self.input_keys:
            if key is None:
                input_list.append(None)
            else:
                key_input = d[key]
                if self.make_batch:
                    key_input = key_input.unsqueeze(0)
                input_list.append(key_input.to(self.device))
        with torch.no_grad():
            d[self.output_key] = self.network(*input_list)
            if self.make_batch:
                # reverse batching
                d[self.output_key] = d[self.output_key][0]
        return d
