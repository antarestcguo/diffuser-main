from ..utils import is_flax_available, is_torch_available

if is_torch_available():
    from .tc_custom_dataset import TCCustomDiffusionDataset, TCCustomDiffusionRMaskInpaintingDataset, \
        TCCustomDiffusionFixedMaskInpaintingDataset,TCCustomDiffusionCleanDataset,TCCustomDiffusionCleanMaskDataset
    from .tc_control_dataset import TCControlDiffusionDataset

from .tc_custom_dataset import random_mask, prepare_mask_and_masked_image, TCCustomDiffusionRMaskInpaintingDataset, \
    TCCustomDiffusionDataset, TCCustomDiffusionFixedMaskInpaintingDataset,TCCustomDiffusionCleanDataset,TCCustomDiffusionCleanMaskDataset
from .tc_control_dataset import TCControlDiffusionDataset
