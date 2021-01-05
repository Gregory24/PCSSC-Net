import numpy as np
from torch.utils.data import Dataset
from vtkplotter import show, Points, Lines


class ColorTemplete:
    def __init__(self):
        self.colors = np.array([
            [255, 128, 128],    # bathtub-0
            [255, 128, 0],      # bed-1
            [128, 255, 128],    # bookshelf-2
            [255, 0, 0],        # cabinet-3
            [255, 255, 255],    # celling-4
            [0, 128, 255],      # chair-5
            [128, 0, 0],        # desk-6
            [255, 255, 0],      # door-7
            [0, 128, 0],        # floor-8
            [0, 0, 0],          # other-9
            [128, 0, 128],      # sink-10
            [255, 0, 255],      # sofa-11
            [128, 128, 255],    # table-12
            [128, 128, 0],      # toilet-13
            [0, 64, 128],       # tv-14
            [128, 255, 255]     # wall-15
        ])


def get_dc_color(dc):
    colors_dc = dc
    colors_dc = np.sum(np.abs(colors_dc), axis=1)
    mx = np.max(colors_dc)
    mn = np.min(colors_dc)
    colors_dc -= (mx + mn) * 0.5
    colors_dc /= (mx - mn)
    colors_dc += 0.5
    colors_dc = -0.5 * np.cos(colors_dc * np.pi) ** 5 + 0.5
    colors_dc = np.vstack((colors_dc, colors_dc, colors_dc))
    colors_dc = colors_dc.transpose()
    return colors_dc


def visualize_all(xyzrgbnordfcli, xyzl, nodes, graph, radius):
    ct = ColorTemplete()
    c_l = np.array([ct.colors[int(xyzrgbnordfcli[i, 12])] for i in range(len(xyzrgbnordfcli))])
    c_dc = get_dc_color(xyzrgbnordfcli[:, 9:12])
    in_component = xyzrgbnordfcli[:, 13].astype('int16')
    n_patches = np.max(in_component) + 1
    c_overseg = np.zeros(shape=(in_component.shape[0], 3))
    seg_colors_tmp = []
    for i in range(n_patches):
        c = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
        seg_colors_tmp.append(c)
    for i in range(in_component.shape[0]):
        c_overseg[i, :] = seg_colors_tmp[in_component[i]]
    c_l_gt = np.array([ct.colors[int(xyzl[i, 3])] for i in range(len(xyzl))])
    pts = Points(xyzrgbnordfcli[:, 0:3], c=xyzrgbnordfcli[:, 3:6], r=radius)
    pts_l = Points(xyzrgbnordfcli[:, 0:3], c=c_l, r=radius)
    pts_dc = Points(xyzrgbnordfcli[:, 0:3], c=c_dc, r=radius)
    pts_overseg = Points(xyzrgbnordfcli[:, 0:3], c=c_overseg, r=radius)
    line_norm = Lines(startPoints=xyzrgbnordfcli[:, 0:3], endPoints=xyzrgbnordfcli[:, 0:3]+xyzrgbnordfcli[:, 6:9], c='red', lw=1, scale=0.075)

    nodes = nodes[:n_patches]
    graph = graph[:n_patches, :n_patches]
    n = graph.shape[0]
    line_graph_start = nodes[:, 0:3]
    line_graph_end_1 = np.zeros((n, 3))
    line_graph_end_2 = np.zeros((n, 3))
    line_graph_end_3 = np.zeros((n, 3))
    line_graph_end_4 = np.zeros((n, 3))
    line_graph_end_5 = np.zeros((n, 3))
    for i in range(n):
        row = graph[i, :]
        idx = np.where(row == 1)[0]
        line_graph_end_1[i, :] = nodes[idx[0], 0:3]
        line_graph_end_2[i, :] = nodes[idx[1], 0:3]
        line_graph_end_3[i, :] = nodes[idx[2], 0:3]
        line_graph_end_4[i, :] = nodes[idx[3], 0:3]
        line_graph_end_5[i, :] = nodes[idx[4], 0:3]
    pts_nodes = Points(nodes[:, 0:3], r=radius)
    line = Lines(startPoints=nodes[:, 0:3], endPoints=nodes[:, 0:3]+nodes[:, 3:6], c='red', lw=2, scale=0.075)
    line_1 = Lines(startPoints=line_graph_start, endPoints=line_graph_end_1, c='green', lw=1)
    line_2 = Lines(startPoints=line_graph_start, endPoints=line_graph_end_2, c='green', lw=1)
    line_3 = Lines(startPoints=line_graph_start, endPoints=line_graph_end_3, c='green', lw=1)
    line_4 = Lines(startPoints=line_graph_start, endPoints=line_graph_end_4, c='green', lw=1)
    line_5 = Lines(startPoints=line_graph_start, endPoints=line_graph_end_5, c='green', lw=1)
    pts_gt = Points(xyzl[:, 0:3], c=c_l_gt, r=radius)
    show([pts, line_norm], at=0, N=6)
    show(pts_l, at=1)
    show(pts_dc, at=2)
    show(pts_overseg, at=3)
    show([pts_nodes, line, line_1, line_2, line_3, line_4, line_5], at=4)
    show(pts_gt, at=5, interactive=1)


class SceneCAD(Dataset):
    def __init__(self, xyz14_txt, nodeinfo_txt, graph_txt, xyzl_txt, max_nodes, data_aug=False):
        self.xyz14_txt = xyz14_txt
        self.nodeinfo_txt = nodeinfo_txt
        self.graph_txt = graph_txt
        self.xyzl_txt = xyzl_txt

        self.xyz14_files = self.load_file_path(self.xyz14_txt)
        self.nodesinfo_files = self.load_file_path(self.nodeinfo_txt)
        self.graph_files = self.load_file_path(self.graph_txt)
        self.xyzl_files = self.load_file_path(self.xyzl_txt)

        self.max_nodes = max_nodes
        self.data_aug = data_aug

    def __len__(self):
        return len(self.xyz14_files)

    def __getitem__(self, index):
        # load data
        xyz14 = np.load(self.xyz14_files[index])
        nodes = np.load(self.nodesinfo_files[index])
        graph = np.load(self.graph_files[index])
        xyzl = np.load(self.xyzl_files[index])
        # data augmentation
        xyz14, nodes, graph, xyzl = self.data_augmentation(xyz14, nodes, graph, xyzl)
        nodes, graph = self.normalize_nodes_graph(nodes, graph)
        return xyz14, nodes, graph, xyzl

    def load_file_path(self, address):
        ''' Load npy files from load.txt
            Input:
                address: string, address of load.txt
            Output:
                file: string
        '''
        file = []
        with open(address, 'r') as f:
            for line in f:
                file.append(line.strip())
        return file

    def data_augmentation(self, xyz14, nodes, graph, xyzl):
        ''' Data augmentation, random rotate at y axis & shuffle order
            Input:
                xyz14: 4096 × 13 numpy
                nodes: n_nodes × 6 numpy
                graph: n_nodes × n_nodes numpy
                xyzl: 4096 × 4 numpy
            Output:
                xyz14: 4096 × 13 numpy
                nodes: n_nodes × 6 numpy
                graph: n_nodes × n_nodes numpy
                xyzl: 4096 × 4 numpy
        '''
        if self.data_aug is True:
            # random rotation
            rotation_angle = np.random.uniform(-1.0 / 6.0, 1.0 / 6.0) * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, -sinval], [0, 1, 0], [sinval, 0, cosval]])
            xyz14[:, 0:3] = np.dot(xyz14[:, 0:3], rotation_matrix)
            xyz14[:, 6:9] = np.dot(xyz14[:, 6:9], rotation_matrix)
            xyz14[:, 9:12] = np.dot(xyz14[:, 9:12], rotation_matrix)
            nodes[:, 0:3] = np.dot(nodes[:, 0:3], rotation_matrix)
            nodes[:, 3:6] = np.dot(nodes[:, 3:6], rotation_matrix)
            nodes[:, 6:9] = np.dot(nodes[:, 6:9], rotation_matrix)
            nodes[:, 9:12] = np.dot(nodes[:, 9:12], rotation_matrix)
            xyzl[:, 0:3] = np.dot(xyzl[:, 0:3], rotation_matrix)
            # shuffle
            np.random.shuffle(xyz14)
            np.random.shuffle(xyzl)
            return xyz14, nodes, graph, xyzl
        else:
            return xyz14, nodes, graph, xyzl

    def normalize_nodes_graph(self, nodes, graph):
        ''' Normalize nodes and graph to a unified size:
            Input:
                nodes: n_nodes × 6 numpy
                graph: n_nodes × n_nodes numpy
            Output:
                nodes: max_nodes × 6 numpy
                graph: max_nodes × 2 * max_nodes numpy
        '''
        n_nodes = nodes.shape[0]
        new_nodes = np.zeros((self.max_nodes, 12))
        new_graph_out = np.zeros((self.max_nodes, self.max_nodes))
        new_nodes[:n_nodes, :] = nodes
        new_graph_out[:n_nodes, :n_nodes] = graph
        new_graph_in = new_graph_out.transpose()
        new_graph = np.hstack((new_graph_out, new_graph_in))
        return new_nodes, new_graph

    def visualize_data(self, index):
        ''' Show input points, leaky groundtruth points and groundtruth points in one window
            Input:
                index: int, index in the dataset
        '''
        xyz14, nodes, graph, xyzl = self.__getitem__(index)
        visualize_all(xyz14, xyzl, nodes, graph, 7)


if __name__ == "__main__":
    train_xyz14_path = '/home/username/dataset/SceneCAD_Graph_v2/Test/xyzrgbnordfcli_fps4096_input.txt'
    train_nodeinfo_path = '/home/username/dataset/SceneCAD_Graph_v2/Test/nodeinfo_fps4096_input.txt'
    train_graph_path = '/home/username/dataset/SceneCAD_Graph_v2/Test/graph_fps4096_input.txt'
    train_xyzl_path = '/home/username/dataset/SceneCAD_Graph_v2/Test/xyzl_fps4096_gt.txt'

    scenecad_train = SceneCAD(train_xyz14_path, train_nodeinfo_path, train_graph_path, train_xyzl_path, 407, data_aug=True)
    xyz14, nodes, graph, xyzl = scenecad_train.__getitem__(95)
