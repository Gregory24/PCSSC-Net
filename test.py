import os
import sys
import importlib
import numpy as np
import torch
import data.scenecad as SceneCAD
import pathmagic_test  # noqa
import chamfer_loss.chamfer_distance_modified as cd
from vtkplotter import show, Points

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))


def visualize_in_pred_gt_4(input_points, xyz, label, ims_xyz, ims_label, gt_points, raduis):
    ''' Show input points, leaky groundtruth points and groundtruth points
        Input:
            input_points: N × 6 numpy
            pred_points: N × 3 numpy
            gt_points_points: N × 4 numpy
    '''
    ct = SceneCAD.ColorTemplete()
    label_color = np.array([ct.colors[int(label[i])] for i in range(len(label))])
    ims_label_color = np.array([ct.colors[int(ims_label[i])] for i in range(len(ims_label))])
    labeled_colors = np.array([ct.colors[int(gt_points[i][3])] for i in range(len(gt_points))])

    pts_input = Points(input_points[:, 0:3], c=input_points[:, 3:6], r=raduis)
    pts_xyz = Points(xyz, c=label_color.tolist(), r=raduis)
    pts_ims_xyz = Points(ims_xyz, c=ims_label_color.tolist(), r=raduis)
    pts_gt = Points(gt_points[:, 0:3], c=labeled_colors, r=raduis)

    show(pts_input, at=0, N=4)
    show(pts_xyz, at=1)
    show(pts_ims_xyz, at=2)
    show(pts_gt, at=3, interactive=1)


def visualize_in_pred_gt_3(input_points, xyz, label, true_label, patch_idx, gt_points, raduis):
    ''' Show input points, leaky groundtruth points and groundtruth points
        Input:
            input_points: N × 6 numpy
            pred_points: N × 3 numpy
            gt_points_points: N × 4 numpy
    '''
    ct = SceneCAD.ColorTemplete()
    label_color = np.array([ct.colors[int(label[i])] for i in range(len(label))])
    true_color = np.array([ct.colors[int(true_label[i])] for i in range(len(true_label))])
    labeled_colors = np.array([ct.colors[int(gt_points[i][3])] for i in range(len(gt_points))])
    seg_colors_tmp = []
    for i in range(patch_idx.max() + 1):
        c = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
        seg_colors_tmp.append(c)
    patch_color = []
    for i in range(patch_idx.shape[0]):
        patch_color.append(seg_colors_tmp[patch_idx[i]])

    pts_input = Points(input_points[:, 0:3], c=input_points[:, 3:6] * 0.5 - 0.5, r=raduis)
    pts_xyz = Points(xyz, c=label_color.tolist(), r=raduis)
    pts_xyz_patch = Points(xyz, c=patch_color, r=raduis)
    pts_true_xyz = Points(xyz, c=true_color.tolist(), r=raduis)
    pts_gt = Points(gt_points[:, 0:3], c=labeled_colors, r=raduis)

    show(pts_input, at=0, N=5, axes=2)
    show(pts_xyz, at=1)
    show(pts_xyz_patch, at=2)
    show(pts_true_xyz, at=3)
    show(pts_gt, at=4, interactive=1)


def visualize_in_pred_gt_2(input_points, xyz, label, true_label, gt_points, raduis):
    ''' Show input points, leaky groundtruth points and groundtruth points
        Input:
            input_points: N × 6 numpy
            pred_points: N × 3 numpy
            gt_points_points: N × 4 numpy
    '''
    ct = SceneCAD.ColorTemplete()
    label_color = np.array([ct.colors[int(label[i])] for i in range(len(label))])
    true_color = np.array([ct.colors[int(true_label[i])] for i in range(len(true_label))])
    labeled_colors = np.array([ct.colors[int(gt_points[i][3])] for i in range(len(gt_points))])

    pts_input = Points(input_points[:, 0:3], c=input_points[:, 3:6] * 0.5 - 0.5, r=raduis)
    pts_xyz = Points(xyz, c=label_color.tolist(), r=raduis)
    pts_true_xyz = Points(xyz, c=true_color.tolist(), r=raduis)
    pts_gt = Points(gt_points[:, 0:3], c=labeled_colors, r=raduis)

    show(pts_input, at=0, N=4, axes=2)
    show(pts_xyz, at=1)
    show(pts_true_xyz, at=2)
    show(pts_gt, at=3, interactive=1)


def visualize_in_pred_gt(input_points, xyz, gt_points):
    ct = SceneCAD.ColorTemplete()
    labeled_colors = np.array([ct.colors[int(gt_points[i][3])] for i in range(len(gt_points))])

    pts_input = Points(input_points[:, 0:3], c=input_points[:, 3:6])
    pts_pred_xyz = Points(xyz)
    pts_gt = Points(gt_points[:, 0:3], c=labeled_colors)

    show(pts_input, at=0, N=3, axes=2)
    show(pts_pred_xyz, at=1)
    show(pts_gt, at=2, interactive=1)


def ims(xyz14, target_num):
    dfc = xyz14[:, 9:12]
    dfc_n = np.sum(dfc * dfc, axis=1)
    idx_sort = np.argsort(-dfc_n)
    idx = idx_sort[:target_num]
    return idx.tolist()


def fps(points, target_num):
    tmp = points[:]
    N = tmp.shape[0]
    if target_num >= N:
        return points

    centroids = np.zeros(target_num).astype('int16')
    distance = np.ones(N) * 1e10
    farthest = int(np.random.randint(0, N))

    for i in range(target_num):
        centroids[i] = farthest
        centroid = tmp[farthest, 0:3]
        dist = np.sum((tmp[:, 0:3] - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
        distance[farthest] = 0
    return centroids


def main():
    test_xyz14_path = '/home/username/dataset/SceneCAD_Graph_v2/Test/xyzrgbnordfcli_fps4096_input.txt'
    test_nodeinfo_path = '/home/username/dataset/SceneCAD_Graph_v2/Test/nodeinfo_fps4096_input.txt'
    test_graph_path = '/home/username/dataset/SceneCAD_Graph_v2/Test/graph_fps4096_input.txt'
    test_xyzl_path = '/home/username/dataset/SceneCAD_Graph_v2/Test/xyzl_fps4096_gt.txt'
    TEST_SET = SceneCAD.SceneCAD(test_xyz14_path, test_nodeinfo_path, test_graph_path, test_xyzl_path, 407, data_aug=False)

    train_xyz14_path = '/home/username/dataset/SceneCAD_Graph_v2/Train/xyzrgbnordfcli_fps4096_input.txt'
    train_nodeinfo_path = '/home/username/dataset/SceneCAD_Graph_v2/Train/nodeinfo_fps4096_input.txt'
    train_graph_path = '/home/username/dataset/SceneCAD_Graph_v2/Train/graph_fps4096_input.txt'
    train_xyzl_path = '/home/username/dataset/SceneCAD_Graph_v2/Train/xyzl_fps4096_gt.txt'
    TRAINING_SET = SceneCAD.SceneCAD(train_xyz14_path, train_nodeinfo_path, train_graph_path, train_xyzl_path, 407, data_aug=False)

    address = os.path.join(PROJECT_DIR, 'log/trained_model')
    dict_address = os.path.join(address, 'checkpoints/best_model.pth')
    sys.path.append(address)

    MODEL = importlib.import_module('pcsscnet')
    network = MODEL.get_model()
    checkpoint = torch.load(dict_address)
    network.load_state_dict(checkpoint['model_state_dict'])
    network = network.cuda()

    xyz14_1, nodes_1, graph_1, xyzl_1 = TEST_SET.__getitem__(232)
    xyz14_2, nodes_2, graph_2, xyzl_2 = TRAINING_SET.__getitem__(11)
    xyz14 = np.stack((xyz14_1, xyz14_2))
    nodes = np.stack((nodes_1, nodes_2))
    graph = np.stack((graph_1, graph_2))
    xyzl = np.stack((xyzl_1, xyzl_1))

    with torch.no_grad():
        xyz12 = torch.Tensor(xyz14[:, :, 0:14]).float().cuda()
        overseg_idx = torch.Tensor(xyz14[:, :, 13]).int().cuda()
        nodes = torch.Tensor(nodes).float().cuda()
        graph = torch.Tensor(graph).float().cuda()
        xyzl = torch.Tensor(xyzl[:, :, 0:3]).cuda()
        network = network.eval()
        pred_xyz, pred_label, true_label, patch_idx, ims_xyz, ims_label = network(xyz12, overseg_idx, nodes, graph)
        xyz = pred_xyz.data.cpu()[0].numpy()
        label = torch.argmax(pred_label[0], dim=1).cpu().numpy().astype('int16')
        im_xyz = ims_xyz.data.cpu()[0].numpy()
        im_label = torch.argmax(ims_label[0], dim=1).cpu().numpy().astype('int16')
        patch_idx = patch_idx.data.cpu()[0].numpy().astype('int16')
        dist = cd.ChamferDistance()
        dist1, dist2, idx = dist(pred_xyz, xyzl)
        dist1_ims, dist2_ims, idx = dist(ims_xyz, xyzl)
        print(f"mean loss: {(torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1))[0]}")
        print(f"mean loss: {(torch.mean(dist1_ims, dim=1) + torch.mean(dist2_ims, dim=1))[0]}")

    visualize_in_pred_gt_4(xyz14_1, xyz, label, im_xyz, im_label, xyzl_1, 8)


if __name__ == "__main__":
    main()
