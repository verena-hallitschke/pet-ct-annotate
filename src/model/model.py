"""Segmentation model."""
import torch


class PETCTAnnotationModel(torch.nn.Module):
    """The PET/CT Annotation model."""

    def __init__(self, c_in=3, c_latent=8):
        super().__init__()
        self.is_proposal = c_in == 2
        self.model = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=c_in,
                out_channels=c_latent,
                kernel_size=3,
                stride=(2, 2, 1),
                padding=(1, 1, 1),
            ),
            torch.nn.BatchNorm3d(num_features=c_latent),
            torch.nn.ReLU(),
            torch.nn.Conv3d(
                in_channels=c_latent,
                out_channels=c_latent * 2,
                kernel_size=3,
                stride=(2, 2, 1),
                padding=(1, 1, 1),
            ),
            torch.nn.BatchNorm3d(num_features=c_latent * 2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(
                in_channels=c_latent * 2,
                out_channels=c_latent,
                kernel_size=3,
                stride=(2, 2, 1),
                padding=(1, 1, 1),
                output_padding=(1, 1, 0),
            ),
            torch.nn.BatchNorm3d(num_features=c_latent),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(
                in_channels=c_latent,
                out_channels=c_latent // 2,
                kernel_size=3,
                stride=(2, 2, 1),
                padding=(1, 1, 1),
                output_padding=(1, 1, 0),
            ),
            torch.nn.BatchNorm3d(num_features=c_latent // 2),
            torch.nn.ReLU(),
            torch.nn.Conv3d(
                in_channels=c_latent // 2,
                out_channels=1,
                kernel_size=3,
                padding=(1, 1, 1),
            ),
        )

    # pylint: disable=too-many-arguments
    def forward(
        self, _, image_ct, image_pet, annotation_fg, annotation_bg, proposal=None
    ):
        """Model pass with an image and (user) annotations."""
        if self.is_proposal:
            x = torch.cat([image_ct, image_pet], dim=1)
        elif proposal is None:
            x = torch.cat([image_ct, image_pet, annotation_fg, annotation_bg], dim=1)
        else:
            x = torch.cat(
                [image_ct, image_pet, annotation_fg, annotation_bg, proposal], dim=1
            )
        x = self.model(x)
        return x
