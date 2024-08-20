import os
from typing import Any, Dict, List
import torch
import torchvision

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class VisslLoader:
    ENV_TORCH_HOME = 'TORCH_HOME'
    ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
    DEFAULT_CACHE_DIR = '~/.cache'
    MODELS = {
        'simclr-rn50': {
            'url': 'https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch',
            'arch': 'resnet50'
        },
        'mocov2-rn50': {
            'url': 'https://dl.fbaipublicfiles.com/vissl/model_zoo/moco_v2_1node_lr.03_step_b32_zero_init/model_final_checkpoint_phase199.torch',
            'arch': 'resnet50'
        },
        'jigsaw-rn50': {
            'url': 'https://dl.fbaipublicfiles.com/vissl/model_zoo/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_17_07_20.db174a43/model_final_checkpoint_phase104.torch',
            'arch': 'resnet50'
        },
        'rotnet-rn50': {
            'url': 'https://dl.fbaipublicfiles.com/vissl/model_zoo/rotnet_rn50_in1k_ep105_rotnet_8gpu_resnet_17_07_20.46bada9f/model_final_checkpoint_phase125.torch',
            'arch': 'resnet50'
        },
        'swav-rn50': {
            'url': 'https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676/model_final_checkpoint_phase799.torch',
            'arch': 'resnet50'
        },
        'pirl-rn50': {
            'url': 'https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl_jigsaw_4node_pirl_jigsaw_4node_resnet_22_07_20.34377f59/model_final_checkpoint_phase799.torch',
            'arch': 'resnet50'
        }
    }

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def _download_and_save_model(self, model_url: str, output_model_filepath: str):
        """
        Downloads the model in vissl format, converts it to torchvision format and
        saves it under output_model_filepath.
        """
        model = load_state_dict_from_url(model_url, map_location=torch.device('cpu'))

        # get the model trunk to rename
        if "classy_state_dict" in model.keys():
            model_trunk = model["classy_state_dict"]["base_model"]["model"]["trunk"]
        elif "model_state_dict" in model.keys():
            model_trunk = model["model_state_dict"]
        else:
            model_trunk = model

        converted_model = self._replace_module_prefix(model_trunk, "_feature_blocks.")
        torch.save(converted_model, output_model_filepath)
        return converted_model

    def _replace_module_prefix(self, state_dict: Dict[str, Any],
                               prefix: str,
                               replace_with: str = ""):
        """
        Remove prefixes in a state_dict needed when loading models that are not VISSL
        trained models.
        Specify the prefix in the keys that should be removed.
        """
        state_dict = {
            (key.replace(prefix, replace_with, 1) if key.startswith(prefix) else key): val
            for (key, val) in state_dict.items()
        }
        return state_dict

    def _get_torch_home(self):
        """
        Gets the torch home folder used as a cache directory for the vissl models.
        """
        torch_home = os.path.expanduser(
            os.getenv(VisslLoader.ENV_TORCH_HOME,
                      os.path.join(os.getenv(VisslLoader.ENV_XDG_CACHE_HOME,
                                             VisslLoader.DEFAULT_CACHE_DIR), 'torch')))
        return torch_home

    def load_model_from_source(self) -> None:
        """
        Load a (pretrained) neural network model from vissl. Downloads the model when it is not available.
        Otherwise, loads it from the cache directory.
        """
        if self.model_name in VisslLoader.MODELS:
            cache_dir = os.path.join(self._get_torch_home(), 'vissl')
            model_filepath = os.path.join(cache_dir, self.model_name + '.torch')
            model_config = VisslLoader.MODELS[self.model_name]
            if not os.path.exists(model_filepath):
                os.makedirs(cache_dir, exist_ok=True)
                model_state_dict = self._download_and_save_model(model_url=model_config['url'],
                                                                 output_model_filepath=model_filepath)
            else:
                model_state_dict = torch.load(model_filepath, map_location=torch.device('cpu'))
            self.model = getattr(torchvision.models, model_config['arch'])()
            self.model.fc = torch.nn.Identity()
            self.model.load_state_dict(model_state_dict, strict=True)
        else:
            raise ValueError(
                f"\nCould not find {self.model_name} among in the Vissl library.\n"
            )
        return self.model
