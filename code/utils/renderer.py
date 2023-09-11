import torch
from einops import rearrange


class VolumeRenderer:
    def __init__(self, args=None):
        self.args = args

    def render(
        self,
        z_vals,
        raw,
        white_bkgd=False,
    ):
        """
        Volume rendering pixels given srdf and radiance of samples

        z_val: z value of each sample, [RN, SN]
        radiance: radiance of each sample, [RN, SN, 3]
        cos_anneal_ratio: cosine annealing ratio
        deviation_network: network to predict deviation
        """

        rgb, sigma = torch.split(raw, [3, 1], dim=-1)

        # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
        # very different scales, and using interval can affect the model's generalization ability.
        # Therefore we don't use the intervals for both training and evaluation.
        sigma2alpha = lambda sigma: 1.0 - torch.exp(-sigma)
        alpha = sigma2alpha(sigma)  # [N_rays, N_samples]
        alpha = rearrange(alpha, "B RN SN 1-> B RN SN")

        # Eq. (3): T
        T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)[
            :, :, :-1
        ]  # [N_rays, N_samples-1]
        T = torch.cat((torch.ones_like(T[:, :, 0:1]), T), dim=-1)  # [N_rays, N_samples]

        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        weights = alpha * T  # [N_rays, N_samples]

        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=2)  # [N_rays, 3]

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - torch.sum(weights, dim=-1, keepdim=True))
        depth_map = torch.sum(weights * z_vals, dim=-1)
        return rgb_map, depth_map, alpha, weights
