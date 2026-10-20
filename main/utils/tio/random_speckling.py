from typing import Optional, Sequence

import torch
import torch.nn.functional as F
import numpy as np
import torchio as tio
from utils import speckle_transform

class RandomSpeckling(tio.Transform):
    """
    TorchIO augmentation that mixes an image with an anisotropically blurred
    version of itself and then applies a random affine intensity transform.

    For each selected channel:

        blurred = GaussianBlur(image)
        output  = scale * (
                     alpha * image
                     + (1 - alpha) * blurred
                  ) + shift

    Parameters
    ----------
    alpha : tuple(float, float)
        Scale factor for intensity increase.
        alpha=1 gives the original image.
        alpha>1 scale a fraction of the pixels by that value.

    fraction : tuple(float, float)
        Fractions to scale.

    sigma : float
        Blur parameter of scalar map.

    channels : sequence[int] or None
        Channels to augment. If None, all channels are augmented.

    p : float
        Probability of applying the transform.

    Notes
    -----
    Assumes image tensors have shape:

        (C, Z, Y, X)

    which is the default TorchIO format.

    Different random parameters are sampled independently for every channel.
    """

    def __init__(
        self,
        alpha=(1.0, 1.5),
        fraction=(0.0, 0.5),
        sigma=2,
        sf=2,
        wavelengths=[[2,3,4],[2,3,4]],
        channels: Optional[Sequence[int]] = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        # check that for whatever alpha is selected, there will exist a 
        # fraction compatible with the transform.
        assert fraction[0] < 1/alpha[1]

        self.alpha = alpha
        self.fraction = fraction
        self.sigma = sigma
        self.sf = sf
        self.ws = wavelengths
        self.channels = channels

    def apply_transform(self, subject: tio.Subject):

        image = subject["image"]
        mask = subject["mask"]
    
        data = image.data.clone()
        targ = mask.data.clone()

        channels = (
            range(data.shape[0])
            if self.channels is None
            else self.channels
        )

        for c in channels:

            alpha = np.random.uniform(*self.alpha)

            fraction_range = [self.fraction[0], min(self.fraction[1], 1/alpha)]
            fraction = np.random.uniform(*fraction_range)
            # fraction = np.random.uniform(*self.range)
    
            channel = data[c].cpu().numpy()
            mask_np = targ.cpu().numpy()

            transformed = speckle_transform(
                channel,
                alpha,
                fraction,
                sigma=self.sigma,
                sf=self.sf,
                wavelengths=self.ws,
                mask=mask_np[0],
            )
    
            data[c] = torch.as_tensor(
                transformed[None],
                dtype=data.dtype,
                device=data.device,
            )

        image.set_data(data)

        return subject