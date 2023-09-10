import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from .utils.grid_sample import grid_sample_2d, grid_sample_3d
from .attention.transformer import LocalFeatureTransformer

import math

PI = math.pi


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=1, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=1, keepdim=True)
    return mean, var


class RayTransformer(nn.Module):
    """
    Ray transformer
    """

    def __init__(self, args, img_feat_dim=32):
        super().__init__()
        self.args = args
        self.geometry_fc_dim = 16
        self.offset = [[0, 0, 0]]
        self.img_feat_dim = img_feat_dim

        self.PE_d_hid = 16

        # transformers
        self.density_ray_transformer = LocalFeatureTransformer(
            d_model=self.geometry_fc_dim + self.PE_d_hid,
            nhead=4,
            layer_names=["self"],
            attention="linear",
        )

        self.softmax = nn.Softmax(dim=-2)
        # to calculate radiance weight
        self.linear_radianceweight_1_softmax = nn.Sequential(
            nn.Linear(self.img_feat_dim + 3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )
        #####################
        self.anti_alias_pooling = args.anti_alias_pooling
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.ray_dir_fc = nn.Sequential(
            nn.Linear(4, 16),
            activation_func,
            nn.Linear(16, self.img_feat_dim + 3),
            activation_func,
        )

        self.base_fc = nn.Sequential(
            nn.Linear((self.img_feat_dim + 3) * 3, 64),
            activation_func,
            nn.Linear(64, 32),
            activation_func,
        )

        self.vis_fc = nn.Sequential(
            nn.Linear(32, 32),
            activation_func,
            nn.Linear(32, 33),
            activation_func,
        )

        self.vis_fc2 = nn.Sequential(
            nn.Linear(32, 32), activation_func, nn.Linear(32, 1), nn.Sigmoid()
        )

        self.geometry_fc = nn.Sequential(
            nn.Linear(32 * 2 + 1, 64),
            activation_func,
            nn.Linear(64, self.geometry_fc_dim),
            activation_func,
        )

        self.out_geometry_fc = nn.Sequential(
            nn.Linear(self.geometry_fc_dim + self.PE_d_hid, 16),
            activation_func,
            nn.Linear(16, 1),
            nn.ReLU(),
        )

        self.rgb_fc = nn.Sequential(
            nn.Linear(32 + 1 + 4, 16),
            activation_func,
            nn.Linear(16, 8),
            activation_func,
            nn.Linear(8, 1),
        )

    def compute_angle(self, point3D, batch_query_camera_T, batch_train_cameras_T):
        """
        Calculates the relative direction between the source views and
        :param point3D: the samples 3D coordinates, shape: (B,RN,SN,3)
        :param batch_query_camera_T: the T component of the extrinsic matrix of the query camera (novel view), shape: (B, 3)
        :param train_cameras: the T components of the extrinsic matrix of the source views cameras, (B, NV, 3)
        :return: (B,NV, RN, SN, 4); The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        """
        B, RN, SN, DimX = point3D.shape
        _, NV, _ = batch_train_cameras_T.shape
        # calculate relative direction (vector_1: direction between camera origin and point, in world coordinates)
        vector_1 = point3D - repeat(batch_query_camera_T, "B DimX -> B 1 1 DimX")
        vector_1 = repeat(vector_1, "B RN SN DimX -> B 1 RN SN DimX")
        # vector_2: relative direction between source views cameras position and 3d points direction, in world coordinates
        vector_2 = point3D.unsqueeze(1) - repeat(
            batch_train_cameras_T, "B L DimX -> B L 1 1 DimX"
        )  # B L RN SN DimX
        vector_1 = vector_1 / torch.linalg.norm(
            vector_1, dim=-1, keepdim=True
        )  # normalize to get direction
        vector_2 = vector_2 / torch.linalg.norm(vector_2, dim=-1, keepdim=True)
        dir_relative = vector_1 - vector_2

        # dot product between the two directions
        vector_3 = torch.bmm(
            rearrange(vector_2, "B NV RN SN DimX -> (B NV RN SN) 1 DimX", DimX=DimX),
            rearrange(
                repeat(vector_1, "B 1 RN SN DimX -> B NV RN SN DimX", NV=NV),
                "B NV RN SN DimX -> (B NV RN SN) DimX 1",
                DimX=3,
            ),
        )
        vector_3 = rearrange(
            vector_3, "(B NV RN SN) 1 1 -> B NV RN SN 1", NV=NV, RN=RN, B=B, SN=SN
        )
        dir_relative = torch.cat([dir_relative, vector_3], dim=-1)
        dir_relative = dir_relative.float()
        return dir_relative

    def order_posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_samples)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table)

        return sinusoid_table

    def forward(self, point3D, batch, source_imgs_feat):
        # B batch size, NV: number of source views
        B, NV, _, H, W = batch["source_imgs"].shape
        # RN: Number of rays, SN: Number of samples per ray
        _, RN, SN, _ = point3D.shape
        # FDim: source views feature
        FDim = source_imgs_feat.size(2)  # feature dim
        CN = len(self.offset)

        # calculate relative direction

        dir_relative = self.compute_angle(
            point3D,
            batch["ref_pose_inv"][
                :, :3, -1
            ],  # T component of the inverse extrinsic camera paramteres, corresponding to the query camera position in the world coords
            batch["source_poses_inv"][:, :, :3, -1],  # source views camera positions
        )

        ########
        ray_diff = dir_relative
        ########

        # -------- project points to feature map
        # B NV RN SN DimXYZ
        point3D = repeat(point3D, "B RN SN DimX -> B NV RN SN DimX", NV=NV).float()
        point3D = torch.cat([point3D, torch.ones_like(point3D[:, :, :, :, :1])], axis=4)

        # B NV 4 4 -> (B NV) 4 4
        points_in_pixel = torch.bmm(
            rearrange(
                batch["source_poses"], "B NV M_1 M_2 -> (B NV) M_1 M_2", M_1=4, M_2=4
            ),
            rearrange(point3D, "B NV RN SN DimX -> (B NV) DimX (RN SN)"),
        )

        points_in_pixel = rearrange(
            points_in_pixel, "(B NV) DimX (RN SN) -> B NV DimX RN SN", B=B, RN=RN
        )
        points_in_pixel = points_in_pixel[:, :, :3]
        # in 2D pixel coordinate
        mask_valid_depth = points_in_pixel[:, :, 2] > 0  # B NV RN SN
        mask_valid_depth = mask_valid_depth.float()
        points_in_pixel = points_in_pixel[:, :, :2] / points_in_pixel[:, :, 2:3]
        # C is source view feature size (FDim)
        img_feat_sampled, mask = grid_sample_2d(
            rearrange(source_imgs_feat, "B NV C H W -> (B NV) C H W"),
            rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2"),
        )
        img_rgb_sampled, _ = grid_sample_2d(
            rearrange(batch["source_imgs"], "B NV C H W -> (B NV) C H W"),
            rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2"),
        )

        mask = rearrange(mask, "(B NV) RN SN -> B NV RN SN", B=B)
        mask = mask * mask_valid_depth
        img_feat_sampled = rearrange(
            img_feat_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B
        )
        img_rgb_sampled = rearrange(
            img_rgb_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B
        )
        direction_feat = self.ray_dir_fc(ray_diff)

        rgb_in = rearrange(img_rgb_sampled, "B NV C RN SN -> B NV RN SN C", B=B)
        rgb_feat = rearrange(img_feat_sampled, "B NV C RN SN -> B NV RN SN C", B=B)
        # Concat the source view features to the rgb values
        rgb_feat = torch.cat([rgb_feat, rgb_in], axis=-1)

        rgb_feat = rgb_feat + direction_feat

        # rgb_feat has the rgb colors, the relative direction and the source view features

        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (
                exp_dot_prod - torch.min(exp_dot_prod, dim=1, keepdim=True)[0]
            ) * mask
            weight = weight / (torch.sum(weight, dim=1, keepdim=True) + 1e-8)
        else:
            # sum over the views
            weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)

        weight = repeat(weight, "B NV RN SN -> B NV RN SN 1")
        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(
            rgb_feat, weight
        )  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)
        x = torch.cat(
            [repeat(globalfeat, "B 1 RN SN DimX -> B NV RN SN DimX", NV=NV), rgb_feat],
            dim=-1,
        )  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)
        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
        mask = repeat(mask, "B NV RN SN -> B NV RN SN 1")
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)
        mean, var = fused_mean_variance(x, weight)

        globalfeat = torch.cat(
            [mean.squeeze(1), var.squeeze(1), weight.mean(dim=1)], dim=-1
        )  # [n_rays, n_samples, 32*2+1]

        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        # number of valid observations over the n views
        num_valid_obs = torch.sum(mask, dim=1)

        globalfeat = rearrange(
            globalfeat, "B RN SN C -> (B RN) SN C", RN=RN, B=B, SN=SN
        )

        globalfeat = torch.cat(
            [
                globalfeat,
                repeat(
                    self.order_posenc(d_hid=self.PE_d_hid, n_samples=SN).type_as(
                        globalfeat
                    ),
                    "SN C -> B_RN SN C",
                    B_RN=B * RN,
                ),
            ],
            axis=2,
        )
        globalfeat = self.density_ray_transformer(globalfeat)
        #######
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma = rearrange(sigma, "(B RN) SN 1 -> B RN SN 1", B=B, RN=RN)
        sigma_out = sigma.masked_fill(
            num_valid_obs < 1, 0.0
        )  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in * blending_weights_valid, dim=1)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        points_in_pixel = rearrange(
            points_in_pixel, "B NV Dim2 RN SN -> B NV RN SN Dim2"
        )
        return out, points_in_pixel
