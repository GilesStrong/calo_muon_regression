from typing import Optional

from torch import Tensor

from lumin.nn.losses.advanced_losses import WeightedFractionalBinnedHuber as LuminWeightedFractionalBinnedHuber  # For backwards compatibility during model loading

__all__ = ['WeightedFractionalBinnedHuber']


class WeightedFractionalBinnedHuber(LuminWeightedFractionalBinnedHuber):
    def __init__(self, perc:float, e_bins:Tensor, mom=0.1, weight:Optional[Tensor]=None):
        super().__init__(perc=perc, bins=e_bins, mom=mom, weight=weight)
