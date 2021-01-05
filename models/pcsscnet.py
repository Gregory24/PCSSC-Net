import numpy as np
import torch
import torch.nn as nn
# from torch.autograd import Variable
import random   # noqa
import pathmagic  # noqa
from models.pointnet import PointNetEncoderSmall
from models.pcssc_models import SC_Module_Small_v2
from models.ggnn import GGNN
from pointnet2_ops.pointnet2_utils import FurthestPointSampling
import chamfer_loss.chamfer_distance_modified as cd
import torch.nn.functional as F



class get_model(nn.Module):
    def __init__(self, sp_size=25, max_node=407):
        super(get_model, self).__init__()
        self.sp_size = sp_size
        self.max_node = max_node
        self.local_pointnet = PointNetEncoderSmall(channel=12)
        self.ggnn = GGNN(512, 262, self.max_node, 30)
        self.sc_decoder = SC_Module_Small_v2()
        self.fps = FurthestPointSampling()
        self.dist = cd.ChamferDistance()

    def forward(self, xyz13, overseg_idx, nodes, graph):
        # PointNet for encoding local feat of each superpoint
        superpoints, num_sp = self.regroup_superpoints(xyz13, overseg_idx)
        superpoints = superpoints.transpose(2, 1).contiguous()
        superfeat = superpoints[:, :13, :]
        superpoints = superpoints[:, :12, :]
        feat_local, feat_point = self.local_pointnet(superpoints)
        # GGNN for message passing and globally encoding
        A = graph
        prop_state, annotation = self.prepare_state_annotation(feat_local, num_sp, nodes)
        feat_nodes, feat_global = self.ggnn(prop_state, annotation, A)

        # Folding decoder
        feat_global = feat_global.view(-1, 1024, 1)
        feat_global = feat_global.repeat(1, 1, self.max_node)
        feat = torch.cat((feat_global, feat_nodes.transpose(2, 1), annotation.transpose(2, 1)[:, :-6, :]), dim=1)
        feat, true_label, patch_idx, feat_dfc, feat_xyz = self.stack_features(feat, feat_point, superfeat, num_sp, 5)
        xyz, label = self.sc_decoder(feat)
        fps_xyz, fps_label, true_label = self.fps_merge(xyz, label, true_label, feat_xyz, 4096)
        return xyz, label, true_label, patch_idx, fps_xyz, fps_label

    def fps_merge(self, xyz, label, true_label, feat_xyz, target_num):
        B = xyz.shape[0]
        points = torch.cat([xyz, feat_xyz], dim=1)
        label = torch.cat([label, label], dim=1)
        true_label = torch.cat([true_label, true_label], dim=1)
        fps_xyz = torch.zeros(B, target_num, 3).float().cuda()
        fps_label = torch.zeros(B, target_num, 16).float().cuda()
        fps_true_label = torch.zeros(B, target_num).float().cuda()
        idx = self.fps_sampling(points, target_num).long().cuda()
        for i in range(B):
            fps_xyz[i, :, :] = points[i, idx[i], :]
            fps_label[i, :, :] = label[i, idx[i], :]
            fps_true_label[i, :] = true_label[i, idx[i]]
        return fps_xyz, fps_label, fps_true_label

    def regroup_superpoints(self, xyz12, overseg_idx):
        B, N, D = xyz12.size()
        xyz12 = xyz12.data.cpu().numpy()
        overseg_idx = overseg_idx.data.cpu().numpy().astype('int16')
        sps = []
        num_sp = []
        for batch_idx in range(B):
            idx = overseg_idx[batch_idx]
            for i in range(idx.max()+1):
                tmp_idx = np.where(idx == i)
                superpoint = xyz12[batch_idx][tmp_idx]
                # superpoint = self.normalize(superpoint)
                sps.append(superpoint)
            num_sp.append(idx.max()+1)
        num_sp = np.array(num_sp).astype('int16').tolist()
        superpoints = []
        for sp in sps:
            if(sp.shape[0] >= self.sp_size):
                sp = torch.Tensor(sp).view(1, -1, D).cuda().contiguous()
                idx = self.fps_sampling(sp[:, :, :3].contiguous(), self.sp_size).long()
                sp = sp[:, idx[0], :]
                superpoints.append(sp)
            else:
                repeat_time = int(self.sp_size / sp.shape[0]) + 1
                sp = np.tile(sp, (repeat_time, 1))
                sp = torch.Tensor(sp[:self.sp_size, :]).view(1, self.sp_size, D).cuda()
                superpoints.append(sp)
        superpoints = torch.cat(superpoints, dim=0)
        return superpoints, num_sp

    def normalize(self, superpoint):
        xyz = superpoint[:, :3]
        centroid = xyz.mean(axis=0)
        cen = np.tile(centroid, (xyz.shape[0], 1))
        superpoint = np.hstack((xyz-cen, superpoint[:, 3:]))
        return superpoint

    def prepare_state_annotation(self, feat_local, num_sp, nodes):
        '''
            input:  feat_local: N_sp * 256
                    batch_size: B * 1
        '''
        B = len(num_sp)
        D = feat_local.shape[1]
        annotation = torch.zeros((B, self.max_node, D+6)).float().cuda()
        prop_state = torch.zeros((B, self.max_node, 512)).float().cuda()
        feat_slice = torch.split(feat_local, num_sp, dim=0)
        for i in range(B):
            annotation[i, :num_sp[i], :D] = feat_slice[i]
            annotation[i, :, D:D+6] = nodes[i, :, 0:6]
            prop_state[i, :num_sp[i], :D+6] = annotation[i, :num_sp[i], :]
        return prop_state, annotation

    def replicate(self, tensor, grid_size):
        B, D, N = tensor.shape
        slice_one = np.ones(N).astype('int16').tolist()
        tensor_sliced = torch.split(tensor, slice_one, dim=2)
        new_tensor_sliced = []
        for i in range(N):
            s = tensor_sliced[i]
            s = s.repeat(1, 1, grid_size * grid_size)
            new_tensor_sliced.append(s)
        tensor = torch.cat(new_tensor_sliced, dim=2)
        return tensor

    def stack_features(self, feat, feat_point, superfeat, num_sp, rep, target_num=4096):
        B, D, _ = feat.shape
        feat_point = torch.cat((feat_point, superfeat), dim=1).transpose(2, 1)       # Nsp * 25 * 77
        feat_point_slice = torch.split(feat_point, num_sp)
        feat_final = []
        for i in range(B):
            feat_p = feat_point_slice[i].reshape(1, -1, 77)
            feat_p = feat_p.transpose(2, 1)
            feat_sp = feat[i, :, :num_sp[i]].reshape(1, D, -1)
            n = feat_sp.shape[2]
            over_idx = torch.Tensor(np.repeat(np.arange(n), rep*rep).reshape(1, 1, n*rep*rep)).cuda()
            feat_sp = self.replicate(feat_sp, rep)
            feat_sp = torch.cat([feat_sp, feat_p, over_idx], dim=1)
            feat_sp = self.importance_sampling(feat_sp, target_num)
            feat_final.append(feat_sp)
        feat_final = torch.cat(feat_final, dim=0)
        true_label = feat_final.transpose(2, 1)[:, :, -2]
        patch_idx = feat_final.transpose(2, 1)[:, :, -1]
        feat_dfc = feat_final[:, -5:-2, :]
        feat_xyz = feat_final[:, -14:-11, :]
        feat_final = feat_final[:, :-14, :]
        return feat_final, true_label, patch_idx, feat_dfc, feat_xyz.transpose(2, 1).contiguous()

    def importance_sampling(self, feat_sp, target_num):
        B, D, N = feat_sp.shape                             # 1, 1869, N
        if (N <= target_num):
            rep = int(target_num / N) + 1
            feat_sp = feat_sp.repeat(1, 1, rep).contiguous()
            return feat_sp[:, :, :target_num]
        else:
            superxyz = feat_sp[:, -13:-10, :].transpose(2, 1).contiguous()
            idx = self.fps_sampling(superxyz, target_num).long().cuda()[0]       # 1 * 4096
            return feat_sp[:, :, idx]

    def fps_sampling(self, xyz, npoints):
        # xyz: B, N, 3
        idx = self.fps.apply(xyz, npoints)      # B, N
        return idx


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.dist = cd.ChamferDistance()

    def forward(self, pred_xyz, pred_label, gt_xyzl, true_label, global_epoch):
        gt_xyz = gt_xyzl[:, :, 0:3]
        gt_label = gt_xyzl[:, :, 3]
        loss_cham, _ = self.chamfer_loss(pred_xyz, gt_xyz)
        loss_seg = self.seg_loss(pred_label, true_label)
        if global_epoch >= 255:
            loss_sem_cham = self.semantic_chamfer_distance(pred_xyz, pred_label, gt_xyz, gt_label)
            loss = 0.02 * loss_cham + loss_sem_cham + loss_seg
        else:
            loss_sem_cham = torch.FloatTensor([1000]).cuda()
            loss = 0.005 * loss_cham + loss_seg
        return loss, loss_cham, loss_sem_cham, loss_seg

    def emd_loss(self, xyz1, xyz2):
        dist2, assigment = self.dist_emd(xyz1, xyz2, 0.005, 50)
        loss_emd = torch.mean(torch.sqrt(dist2))
        return loss_emd

    def chamfer_loss(self, xyz1, xyz2):
        dist1, dist2, idx = self.dist(xyz1, xyz2)
        # idx = self.dist.nn_idx1
        loss = torch.mean(torch.sum(dist1, dim=1) + torch.sum(dist2, dim=1))
        return loss, idx

    def mean_chamfer_loss(self, xyz1, xyz2):
        dist1, dist2, idx = self.dist(xyz1, xyz2)
        loss = torch.mean(dist1) + torch.mean(dist2)
        return loss

    def seg_loss(self, pred_label, true_label):
        true_label = true_label.long()
        pred_label = pred_label.contiguous().view(-1, 16)
        true_label = true_label.view(-1, 1)[:, 0]
        loss = F.nll_loss(pred_label, true_label)
        return loss

    def semantic_chamfer_distance(self, pred_xyz, pred_label, gt_xyz, gt_label):
        B, _, _ = pred_xyz.shape
        loss_cham = torch.zeros(16).cuda()
        times_cham = torch.zeros(16).cuda()
        weights = np.array([1, 1, 1, 1, 1, 0.01, 1, 1, 1, 0.01, 1, 1, 1, 1, 1, 1])
        class_weights = torch.FloatTensor(weights)
        for i in range(B):
            pred_l = pred_label[i, :, :].data.cpu().numpy().astype('int16').argmax(axis=1)      # 4096
            gt_l = gt_label[i, :].data.cpu().numpy().astype('int16')                            # 4096
            for j in range(16):
                # if j == 9:      # without 'other' class
                #     continue
                idx_xyz = (pred_l == j)
                idx_gt = (gt_l == j)
                xyz = pred_xyz[i, idx_xyz, :]
                gt = gt_xyz[i, idx_gt, :]
                if xyz.shape[0] != 0 and gt.shape[0] != 0:
                    xyz = xyz.view(1, -1, 3).contiguous()
                    gt = gt.view(1, -1, 3).contiguous()
                    dist = self.mean_chamfer_loss(xyz, gt)
                    loss_cham[j] += (dist * class_weights[j])
                    times_cham[j] += 1
        idx_non_zero = (loss_cham != 0)
        loss_cham = loss_cham[idx_non_zero]
        times_cham = times_cham[idx_non_zero]
        loss_cham = loss_cham / times_cham
        loss = loss_cham.sum()
        return loss
