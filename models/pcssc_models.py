import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSegMLP(nn.Module):
    def __init__(self):
        super(GraphSegMLP, self).__init__()
        self.mlp1 = nn.Linear(256, 128)
        self.mlp2 = nn.Linear(128, 16)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, feat_nodes):
        x = F.relu(self.bn1(self.mlp1(feat_nodes)))
        x = self.mlp2(x)
        return x


class FoldingBasedDecoder(nn.Module):
    def __init__(self, ch_in=1286):
        super(FoldingBasedDecoder, self).__init__()
        # First folding
        self.conv1_1 = nn.Conv1d(ch_in, 1024, 1)
        self.conv1_2 = nn.Conv1d(1024, 512, 1)
        self.conv1_3 = nn.Conv1d(512, 256, 1)
        self.conv1_4 = nn.Conv1d(256, 64, 1)
        self.conv1_5 = nn.Conv1d(64, 3, 1)
        self.bn1_1 = nn.BatchNorm1d(1024)
        self.bn1_2 = nn.BatchNorm1d(512)
        self.bn1_3 = nn.BatchNorm1d(256)
        self.bn1_4 = nn.BatchNorm1d(64)
        # Second folding
        self.conv2_1 = nn.Conv1d(ch_in+3, 1024, 1)
        self.conv2_2 = nn.Conv1d(1024, 512, 1)
        self.conv2_3 = nn.Conv1d(512, 256, 1)
        self.conv2_4 = nn.Conv1d(256, 64, 1)
        self.conv2_5 = nn.Conv1d(64, 3, 1)
        self.bn2_1 = nn.BatchNorm1d(1024)
        self.bn2_2 = nn.BatchNorm1d(512)
        self.bn2_3 = nn.BatchNorm1d(256)
        self.bn2_4 = nn.BatchNorm1d(64)

    def forward(self, sp):
        # sp: B, D, N
        # First folding
        x = F.relu(self.bn1_1(self.conv1_1(sp)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn1_3(self.conv1_3(x)))
        x = F.relu(self.bn1_4(self.conv1_4(x)))
        x = self.conv1_5(x)
        # Second folding
        y = torch.cat((sp, x), dim=1)
        y = F.relu(self.bn2_1(self.conv2_1(y)))
        y = F.relu(self.bn2_2(self.conv2_2(y)))
        y = F.relu(self.bn2_3(self.conv2_3(y)))
        y = F.relu(self.bn2_4(self.conv2_4(y)))
        y = self.conv2_5(y)
        y = y.transpose(2, 1).contiguous()
        return y    # B, N, 3


class FoldingBasedLocalDecoder(nn.Module):
    def __init__(self, ch_in=1286):
        super(FoldingBasedLocalDecoder, self).__init__()
        # First folding
        self.conv1 = nn.Conv1d(ch_in, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 128, 1)
        self.conv4 = nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.drop3 = nn.Dropout(0.1)
        # Second folding
        self.conv5 = nn.Conv1d(256+64+3, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, 64, 1)
        self.conv8 = nn.Conv1d(64, 3, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(64)
        self.drop4 = nn.Dropout(0.1)
        self.drop5 = nn.Dropout(0.1)
        self.drop6 = nn.Dropout(0.1)

    def forward(self, sp):
        # sp: B, D, N
        # First folding
        x = self.drop1(F.relu(self.bn1(self.conv1(sp))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = self.conv4(x)
        # Second folding
        y = torch.cat((sp[:, -320:], x), dim=1)
        y = self.drop4(F.relu(self.bn4(self.conv5(y))))
        y = self.drop5(F.relu(self.bn5(self.conv6(y))))
        y = self.drop6(F.relu(self.bn6(self.conv7(y))))
        y = self.conv8(y)
        y = y.transpose(2, 1).contiguous()
        return y    # B, N, 3


class FoldingBasedLocalDecoderNonDrop(nn.Module):
    def __init__(self, ch_in=1286):
        super(FoldingBasedLocalDecoderNonDrop, self).__init__()
        # First folding
        self.conv1 = nn.Conv1d(ch_in, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 128, 1)
        self.conv4 = nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)
        # Second folding
        self.conv5 = nn.Conv1d(256+64+3, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, 64, 1)
        self.conv8 = nn.Conv1d(64, 3, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(64)

    def forward(self, sp):
        # sp: B, D, N
        # First folding
        x = F.relu(self.bn1(self.conv1(sp)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        # Second folding
        y = torch.cat((sp[:, -320:], x), dim=1)
        y = F.relu(self.bn4(self.conv5(y)))
        y = F.relu(self.bn5(self.conv6(y)))
        y = F.relu(self.bn6(self.conv7(y)))
        y = self.conv8(y)
        y = y.transpose(2, 1).contiguous()
        return y    # B, N, 3


class FoldingDecoderLocal(nn.Module):
    def __init__(self, ch_in=1286):
        super(FoldingDecoderLocal, self).__init__()
        # First folding
        self.conv1 = nn.Conv1d(ch_in, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 128, 1)
        self.conv4 = nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)
        # Second folding
        self.conv5 = nn.Conv1d(512+256+64+3, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, 64, 1)
        self.conv8 = nn.Conv1d(64, 3, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(64)

    def forward(self, sp):
        # sp: B, D, N
        # First folding
        x = F.relu(self.bn1(self.conv1(sp)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        # Second folding
        y = torch.cat((sp[:, -832:, :], x), dim=1)
        y = F.relu(self.bn4(self.conv5(y)))
        y = F.relu(self.bn5(self.conv6(y)))
        y = F.relu(self.bn6(self.conv7(y)))
        y = self.conv8(y)
        y = y.transpose(2, 1).contiguous()
        return y    # B, N, 3


class Single_FoldingBasedDecoder(nn.Module):
    def __init__(self, ch_in=1286):
        super(Single_FoldingBasedDecoder, self).__init__()
        self.conv1 = nn.Conv1d(ch_in, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.conv4 = nn.Conv1d(256, 3, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.drop3 = nn.Dropout(0.1)

    def forward(self, sp):
        # sp: B, D, N
        # First folding
        x = self.drop1(F.relu(self.bn1(self.conv1(sp))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        return x    # B, N, 3


class FoldingDecoderSmall(nn.Module):
    def __init__(self, ch_in=1024+512+256+64):
        super(FoldingDecoderSmall, self).__init__()
        # First folding
        self.conv1 = nn.Conv1d(ch_in, 512, 1)
        self.conv2 = nn.Conv1d(512, 128, 1)
        self.conv3 = nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        # Second folding
        self.conv4 = nn.Conv1d(512+256+64+3, 512, 1)
        self.conv5 = nn.Conv1d(512, 128, 1)
        self.conv6 = nn.Conv1d(128, 3, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(128)

    def forward(self, sp):
        # sp: B, D, N
        # First folding
        x = F.relu(self.bn1(self.conv1(sp)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        # Second folding
        y = torch.cat((sp[:, -(512+256+64):, :], x), dim=1)
        y = F.relu(self.bn3(self.conv4(y)))
        y = F.relu(self.bn4(self.conv5(y)))
        y = self.conv6(y)
        y = y.transpose(2, 1).contiguous()
        return y    # B, N, 3


class SC_Module(nn.Module):
    def __init__(self, ch_in_seg=1289, dim_th=336):
        super(SC_Module, self).__init__()
        self.sc_seg = SC_Seg(ch_in=ch_in_seg)
        self.sc_comp = SC_Comp(ch_in=ch_in_seg+16, dim_th=336)

    def forward(self, feat):
        B, D, N = feat.shape
        feat_seg = feat                                     # B * 1289 * N
        f_seg = self.sc_seg(feat_seg)                       # B * 16 * N
        feat_comp = torch.cat((feat, f_seg), dim=1)         # B * 1305 * N
        f_comp = self.sc_comp(feat_comp)                    # B * N * 3

        f_seg = f_seg.transpose(2, 1).contiguous()          # B * N * 16
        f_seg = F.log_softmax(f_seg.view(-1, 16), dim=-1)   # (B * N) * 16
        f_seg = f_seg.view(B, N, 16)
        return f_comp, f_seg                                # B * N * 3, B * N * 16


class SC_Seg(nn.Module):
    def __init__(self, ch_in=1289):
        super(SC_Seg, self).__init__()
        self.conv1 = nn.Conv1d(ch_in, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.conv4 = nn.Conv1d(256, 64, 1)
        self.conv5 = nn.Conv1d(64, 16, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, feat_seg):
        x = F.relu(self.bn1(self.conv1(feat_seg)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x


class SC_Comp(nn.Module):
    def __init__(self, ch_in=1305, dim_th=336):
        super(SC_Comp, self).__init__()
        self.dim_th = dim_th
        # first folding
        self.conv1_1 = nn.Conv1d(ch_in, 1024, 1)
        self.conv1_2 = nn.Conv1d(1024, 512, 1)
        self.conv1_3 = nn.Conv1d(512, 128, 1)
        self.conv1_4 = nn.Conv1d(128, 3, 1)
        self.bn1_1 = nn.BatchNorm1d(1024)
        self.bn1_2 = nn.BatchNorm1d(512)
        self.bn1_3 = nn.BatchNorm1d(128)
        # Second folding
        self.conv2_1 = nn.Conv1d(dim_th+3, 256, 1)
        self.conv2_2 = nn.Conv1d(256, 128, 1)
        self.conv2_3 = nn.Conv1d(128, 64, 1)
        self.conv2_4 = nn.Conv1d(64, 3, 1)
        self.bn2_1 = nn.BatchNorm1d(256)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.bn2_3 = nn.BatchNorm1d(64)

    def forward(self, feat_comp):
        # First folding
        x = F.relu(self.bn1_1(self.conv1_1(feat_comp)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn1_3(self.conv1_3(x)))
        x = self.conv1_4(x)
        # Second folding
        y = torch.cat((feat_comp[:, -self.dim_th:, :], x), dim=1)
        y = F.relu(self.bn2_1(self.conv2_1(y)))
        y = F.relu(self.bn2_2(self.conv2_2(y)))
        y = F.relu(self.bn2_3(self.conv2_3(y)))
        y = self.conv2_4(y)
        return y.transpose(2, 1).contiguous()


class SinglePointwiseDecoderSmall(nn.Module):
    def __init__(self, ch_in=1024+512+256+64):
        super(SinglePointwiseDecoderSmall, self).__init__()
        self.conv1 = nn.Conv1d(ch_in, 512, 1)
        self.conv2 = nn.Conv1d(512, 128, 1)
        self.conv3 = nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, sp):
        # sp: B, D, N
        x = F.relu(self.bn1(self.conv1(sp)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


class SC_Module_Small(nn.Module):
    def __init__(self, ch_in=1024+512+256+64, ch_in_seg=1024+512+256+64+3, dim_th=512+256+64+16):
        super(SC_Module_Small, self).__init__()
        self.single_decoder = SinglePointwiseDecoderSmall(ch_in=ch_in)
        self.sc_seg = SC_Seg_Small(ch_in=ch_in_seg)
        self.sc_comp = SC_Comp_Small(ch_in=ch_in_seg+16, dim_th=dim_th)

    def forward(self, feat):
        B, D, N = feat.shape
        tmp_xyz = self.single_decoder(feat)
        feat_seg = torch.cat((feat, tmp_xyz), dim=1)
        f_seg = self.sc_seg(feat_seg)
        feat_comp = torch.cat((feat, f_seg), dim=1)
        f_comp = self.sc_comp(feat_comp, tmp_xyz)

        f_seg = f_seg.transpose(2, 1).contiguous()
        f_seg = F.log_softmax(f_seg.view(-1, 16), dim=-1)
        f_seg = f_seg.view(B, N, 16)
        return f_comp, f_seg


class SC_Seg_Small(nn.Module):
    def __init__(self, ch_in=1024+512+256+64):
        super(SC_Seg_Small, self).__init__()
        self.conv1 = nn.Conv1d(ch_in, 512, 1)
        self.conv2 = nn.Conv1d(512, 128, 1)
        self.conv3 = nn.Conv1d(128, 16, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(16)

    def forward(self, feat_seg):
        x = F.relu(self.bn1(self.conv1(feat_seg)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x


class SC_Comp_Small(nn.Module):
    def __init__(self, ch_in=1305, dim_th=512+256+64+16):
        super(SC_Comp_Small, self).__init__()
        self.dim_th = dim_th
        self.conv1 = nn.Conv1d(dim_th+3, 512, 1)
        self.conv2 = nn.Conv1d(512, 128, 1)
        self.conv3 = nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, feat_comp, tmp_xyz):
        feat = feat_comp[:, -self.dim_th:, :]
        y = torch.cat([feat, tmp_xyz], dim=1)
        y = F.relu(self.bn1(self.conv1(y)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.conv3(y)
        return y.transpose(2, 1).contiguous()


class SC_Comp_Small_G(nn.Module):
    def __init__(self, ch_in=1305, dim_th=1024+512+256+64+16):
        super(SC_Comp_Small_G, self).__init__()
        self.dim_th = dim_th
        self.conv1 = nn.Conv1d(dim_th+3, 512, 1)
        self.conv2 = nn.Conv1d(512, 128, 1)
        self.conv3 = nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, feat_comp, tmp_xyz):
        feat = feat_comp[:, :, :]
        y = torch.cat([feat, tmp_xyz], dim=1)
        y = F.relu(self.bn1(self.conv1(y)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.conv3(y)
        return y.transpose(2, 1).contiguous()


class SC_Module_Small_v2(nn.Module):
    def __init__(self, ch_in_seg=1024+512+256+64):
        super(SC_Module_Small_v2, self).__init__()
        self.single_decoder = SinglePointwiseDecoderSmall(ch_in=ch_in_seg)
        self.sc_seg = SC_Seg_Small(ch_in=ch_in_seg)
        self.sc_comp = SC_Comp_Small(ch_in=ch_in_seg+16, dim_th=512+256+64+16)

    def forward(self, feat):
        B, D, N = feat.shape
        f_comp = self.single_decoder(feat)
        f_seg = self.sc_seg(feat)
        feat_comp = torch.cat((feat, f_seg), dim=1)
        f_comp_refine = self.sc_comp(feat_comp, f_comp)

        f_seg = f_seg.transpose(2, 1).contiguous()
        f_seg = F.log_softmax(f_seg.view(-1, 16), dim=-1)
        f_seg = f_seg.view(B, N, 16)
        return f_comp_refine, f_seg


class SC_Module_Small_v2_G(nn.Module):
    def __init__(self, ch_in_seg=1024+512+256+64):
        super(SC_Module_Small_v2_G, self).__init__()
        self.single_decoder = SinglePointwiseDecoderSmall(ch_in=ch_in_seg)
        self.sc_seg = SC_Seg_Small(ch_in=ch_in_seg)
        self.sc_comp = SC_Comp_Small_G(ch_in=ch_in_seg+16, dim_th=1024+512+256+64+16)

    def forward(self, feat):
        B, D, N = feat.shape
        f_comp = self.single_decoder(feat)
        f_seg = self.sc_seg(feat)
        feat_comp = torch.cat((feat, f_seg), dim=1)
        f_comp_refine = self.sc_comp(feat_comp, f_comp)

        f_seg = f_seg.transpose(2, 1).contiguous()
        f_seg = F.log_softmax(f_seg.view(-1, 16), dim=-1)
        f_seg = f_seg.view(B, N, 16)
        return f_comp_refine, f_seg


class SC_Module_Small_v2_Seg(nn.Module):
    def __init__(self, ch_in_seg=1024+512+256+64):
        super(SC_Module_Small_v2_Seg, self).__init__()
        self.sc_seg = SC_Seg_Small(ch_in=ch_in_seg)

    def forward(self, feat):
        B, D, N = feat.shape
        f_seg = self.sc_seg(feat)

        f_seg = f_seg.transpose(2, 1).contiguous()
        f_seg = F.log_softmax(f_seg.view(-1, 16), dim=-1)
        f_seg = f_seg.view(B, N, 16)
        return f_seg


class SC_Module_Small_v2_Comp(nn.Module):
    def __init__(self, ch_in_seg=1024+512+256+64):
        super(SC_Module_Small_v2_Comp, self).__init__()
        self.single_decoder = SinglePointwiseDecoderSmall(ch_in=ch_in_seg)
        self.sc_comp = SC_Comp_Small(ch_in=ch_in_seg, dim_th=512+256+64)

    def forward(self, feat):
        B, D, N = feat.shape
        f_comp = self.single_decoder(feat)
        f_comp_refine = self.sc_comp(feat, f_comp)

        return f_comp_refine


class PointNetRes(nn.Module):
    def __init__(self):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)

    def forward(self, x):
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x)
        return x


class SC_Module_Small_v2_Refine(nn.Module):
    def __init__(self, ch_in_seg=1024+512+256+64):
        super(SC_Module_Small_v2_Refine, self).__init__()
        self.single_decoder = SinglePointwiseDecoderSmall(ch_in=ch_in_seg)
        self.sc_seg = SC_Seg_Small(ch_in=ch_in_seg)
        self.sc_comp = SC_Comp_Small(ch_in=ch_in_seg+16, dim_th=512+256+64+16)
        self.sc_refine = PointNetRes()

    def forward(self, feat, input_xyz):
        B, D, N = feat.shape
        # segmentation
        f_seg = self.sc_seg(feat)
        # completion
        feat_3072 = feat[:, :, :3072]
        f_comp = self.single_decoder(feat_3072)
        f_seg_3072 = f_seg[:, :, :3072]
        feat_comp = torch.cat((feat_3072, f_seg_3072), dim=1)
        f_comp = self.sc_comp(feat_comp, f_comp)
        # residual refine
        f_comp_4096 = torch.cat([f_comp.transpose(2, 1).contiguous(), input_xyz], dim=2)
        delta = self.sc_refine(f_comp_4096)
        f_comp_refine = f_comp_4096 + delta

        f_seg = f_seg.transpose(2, 1).contiguous()
        f_seg = F.log_softmax(f_seg.view(-1, 16), dim=-1)
        f_seg = f_seg.view(B, N, 16)
        f_comp_refine = f_comp_refine.transpose(2, 1).contiguous()
        return f_comp, f_comp_refine, f_seg
