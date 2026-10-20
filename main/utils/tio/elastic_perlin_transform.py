import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio

from utils.perlin_noise import perlin_optim_3d


class ElasticPerlinTransform(tio.Transform):
    """
    Elastic deformation driven by Perlin-noise displacement fields.

    A smooth random displacement field is generated independently for
    each spatial axis and applied to all images in the subject using
    PyTorch's grid_sample.

    Parameters
    ----------
    grid_points : int or sequence[int]
        Perlin-noise resolution.

    p_amplitude : float
        Maximum displacement as a fraction of image size.

    sf : int
        Perlin optimisation factor.

    image_interpolation : str
        Interpolation for ScalarImage data.

    p : float
        Probability of applying the transform.
    """

    def __init__(
        self,
        grid_points=8,
        p_amplitude=0.05,
        sf=2,
        image_interpolation="bilinear",
        p=0.25,
    ):
        super().__init__(p=p)

        self.grid_points = grid_points
        self.p_amplitude = p_amplitude
        self.sf = sf
        self.image_interpolation = image_interpolation

    @staticmethod
    def _norm(x):
        return (x - x.min()) / (x.max() - x.min())

    @staticmethod
    def _next_power_of_two(x):
        if (x & (x - 1)) == 0:
            return x
        return 1 << (x - 1).bit_length()

    def _make_displacement(self, shape):

        zres, yres, xres = shape

        if isinstance(self.grid_points, int):
            gp = [self.grid_points] * 3
        else:
            gp = self.grid_points

        res = [
            self._next_power_of_two(zres),
            self._next_power_of_two(yres),
            self._next_power_of_two(xres),
        ]

        dz = (
            2
            * self.p_amplitude
            * zres
            * (self._norm(perlin_optim_3d(res, gp, sf=self.sf)) - 0.5)
        )

        dy = (
            2
            * self.p_amplitude
            * yres
            * (self._norm(perlin_optim_3d(res, gp, sf=self.sf)) - 0.5)
        )

        dx = (
            2
            * self.p_amplitude
            * xres
            * (self._norm(perlin_optim_3d(res, gp, sf=self.sf)) - 0.5)
        )

        return (
            dz[:zres, :yres, :xres],
            dy[:zres, :yres, :xres],
            dx[:zres, :yres, :xres],
        )

    def apply_transform(self, subject):

        reference_image = next(
            iter(subject.get_images(intensity_only=False))
        )

        _, zres, yres, xres = reference_image.data.shape

        dz, dy, dx = self._make_displacement(
            (zres, yres, xres)
        )

        device = reference_image.data.device
        dtype = reference_image.data.dtype

        z = torch.linspace(-1, 1, zres, device=device)
        y = torch.linspace(-1, 1, yres, device=device)
        x = torch.linspace(-1, 1, xres, device=device)

        zz, yy, xx = torch.meshgrid(
            z, y, x, indexing="ij"
        )

        dz = torch.tensor(dz, device=device, dtype=dtype)
        dy = torch.tensor(dy, device=device, dtype=dtype)
        dx = torch.tensor(dx, device=device, dtype=dtype)

        dx = dx * (2.0 / max(xres - 1, 1))
        dy = dy * (2.0 / max(yres - 1, 1))
        dz = dz * (2.0 / max(zres - 1, 1))

        grid = torch.stack(
            [
                xx + dx,
                yy + dy,
                zz + dz,
            ],
            dim=-1,
        )

        grid = grid.unsqueeze(0)

        for image in subject.get_images(intensity_only=False):

            data = image.data.unsqueeze(0)

            if isinstance(image, tio.LabelMap):
                mode = "nearest"
            else:
                mode = self.image_interpolation

            transformed = F.grid_sample(
                data,
                grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            )

            image.set_data(transformed.squeeze(0))

        return subject