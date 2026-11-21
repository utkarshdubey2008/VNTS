"""
Base module and tensor management for VNTS (Vector Nullspace Transformation System).
Strictly zero-gradient, zero-autograd architecture.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch


class VNTSModule:
    """
    Abstract base class for all VNTS neural components.
    Guarantees no autograd graphs, pure tensor operations, and clean parameter handling.
    """
    def __init__(self):
        self._parameters: Dict[str, torch.Tensor] = {}
        self._buffers: Dict[str, torch.Tensor] = {}
        self._submodules: Dict[str, 'VNTSModule'] = {}
        self.device = torch.device("cpu")

    def register_parameter(self, name: str, tensor: Optional[torch.Tensor]):
        if tensor is not None:
            # Strictly enforce zero gradients
            tensor = tensor.detach().clone()
            tensor.requires_grad = False
            self._parameters[name] = tensor
        else:
            self._parameters.pop(name, None)

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor]):
        if tensor is not None:
            tensor = tensor.detach().clone()
            tensor.requires_grad = False
            self._buffers[name] = tensor
        else:
            self._buffers.pop(name, None)

    def register_module(self, name: str, module: 'VNTSModule'):
        self._submodules[name] = module

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        for k, v in self._parameters.items():
            self._parameters[k] = v.to(self.device)
        for k, v in self._buffers.items():
            self._buffers[k] = v.to(self.device)
        for sub in self._submodules.values():
            sub.to(self.device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "params": {k: v.cpu().clone() for k, v in self._parameters.items()},
            "buffers": {k: v.cpu().clone() for k, v in self._buffers.items()},
            "submodules": {k: sub.state_dict() for k, sub in self._submodules.items()}
        }
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        for k, v in state.get("params", {}).items():
            if k in self._parameters:
                self._parameters[k] = v.to(self.device)
        for k, v in state.get("buffers", {}).items():
            if k in self._buffers:
                self._buffers[k] = v.to(self.device)
        for k, sub_state in state.get("submodules", {}).items():
            if k in self._submodules:
                self._submodules[k].load_state_dict(sub_state)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")
