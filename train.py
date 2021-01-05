import os
import sys
import argparse
import logging
import time
import datetime
import importlib
import shutil
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
import data.scenecad as SceneCAD


# Project directory
PROJECT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(PROJECT_DIR, 'models'))


# Arguments declearation
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pcsscnet', help='model name [default: pointnet2]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=1500, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=4096, help='Point Number [default: 1000]')
    parser.add_argument('--step_size', type=int,  default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # cuda
    # torch.backends.cudnn.enabled = False  # disable cudnn for CUDNN_NOT_SUPPORT, may not contiguous
    # ---------------- Create log dir -------------------
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('pcsscnet')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # ---------------- Global params -------------------
    # CHECK_RADIUS = 0.05
    # NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    # ---------------- Training set --------------------
    train_xyz14_path = '/home/username/dataset/SceneCAD_Graph_v2/Train/xyzrgbnordfcli_fps4096_input.txt'
    train_nodeinfo_path = '/home/username/dataset/SceneCAD_Graph_v2/Train/nodeinfo_fps4096_input.txt'
    train_graph_path = '/home/username/dataset/SceneCAD_Graph_v2/Train/graph_fps4096_input.txt'
    train_xyzl_path = '/home/username/dataset/SceneCAD_Graph_v2/Train/xyzl_fps4096_gt.txt'

    TRAINING_SET = SceneCAD.SceneCAD(train_xyz14_path, train_nodeinfo_path, train_graph_path, train_xyzl_path, 407, data_aug=True)
    trainDataLoader = torch.utils.data.DataLoader(TRAINING_SET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=lambda x: np.random.seed(x+int(time.time())))
    # train_weight = torch.Tensor(TRAINING_SET.train_weight).cuda()     # cuda
    test_xyz14_path = '/home/username/dataset/SceneCAD_Graph_v2/Test/xyzrgbnordfcli_fps4096_input.txt'
    test_nodeinfo_path = '/home/username/dataset/SceneCAD_Graph_v2/Test/nodeinfo_fps4096_input.txt'
    test_graph_path = '/home/username/dataset/SceneCAD_Graph_v2/Test/graph_fps4096_input.txt'
    test_xyzl_path = '/home/username/dataset/SceneCAD_Graph_v2/Test/xyzl_fps4096_gt.txt'

    TEST_SET = SceneCAD.SceneCAD(test_xyz14_path, test_nodeinfo_path, test_graph_path, test_xyzl_path, 407, data_aug=False)
    testDataLoader = torch.utils.data.DataLoader(TEST_SET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=lambda x: np.random.seed(x+int(time.time())))
    # ---------------- Log dataset info --------------------
    log_string("The number of training data is: %d" % len(TRAINING_SET))

    # ---------------- Config network model --------------------
    MODEL = importlib.import_module(args.model)
    shutil.copy(os.path.join(PROJECT_DIR, 'models/%s.py' % args.model), str(experiment_dir))

    network = MODEL.get_model().cuda()    # cuda
    criterion = MODEL.get_loss().cuda()   # cuda

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Conv1d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    start_epoch = 0
    network = network.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(network.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_loss = 1000
    best_comp = 1000
    best_smcomp = 1000
    best_seg = 1000
    # ---------------- Start training --------------------
    for epoch in range(start_epoch, args.epoch):
        '''Train on scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        network = network.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)

        loss_sum = 0
        loss_comp = 0
        loss_smcomp = 0
        loss_seg = 0
        # ---------------- Start batch set training --------------------
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            xyz14, nodes, graph, xyzl = data
            xyz12 = xyz14[:, :, :12]
            overseg_idx = xyz14[:, :, 13]
            xyz14 = xyz14.float().cuda()
            xyz12, nodes, graph, overseg_idx, xyzl = xyz12.float().cuda(), nodes.float().cuda(), graph.float().cuda(), overseg_idx.int().cuda(), xyzl.float().cuda()  # cuda
            # train network
            optimizer.zero_grad()
            network = network.train()
            pred_xyz, pred_label, true_label, patch_idx, m_xyz, m_label = network(xyz14, overseg_idx, nodes, graph)
            # back probagation
            loss, lcomp, lsmcomp, lseg = criterion(pred_xyz, pred_label, xyzl, true_label, global_epoch)
            loss.backward()
            optimizer.step()

            loss_sum += loss
            loss_comp += lcomp
            loss_smcomp += lsmcomp
            loss_seg += lseg
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training loss cham: %f' % (loss_comp / num_batches))
        log_string('Training loss sem cham: %f' % (loss_smcomp / num_batches))
        log_string('Training loss seg: %f' % (loss_seg / num_batches))

        if (epoch+1) % 100 == 0:
            logger.info('Save model...')
            savepath = (str(checkpoints_dir) + '/%d_model.pth' % global_epoch)
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        with torch.no_grad():
            num_batches = len(testDataLoader)
            loss_sum = 0
            loss_comp = 0
            loss_smcomp = 0
            loss_seg = 0
            log_string('---- EPOCH %03d TEST ----' % (global_epoch + 1))
            for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                xyz14, nodes, graph, xyzl = data
                xyz12 = xyz14[:, :, :12]
                overseg_idx = xyz14[:, :, 13]
                xyz14 = xyz14.float().cuda()
                xyz12, nodes, graph, overseg_idx, xyzl = xyz12.float().cuda(), nodes.float().cuda(), graph.float().cuda(), overseg_idx.int().cuda(), xyzl.float().cuda()  # cuda
                network = network.eval()
                pred_xyz, pred_label, true_label, patch_idx, m_xyz, m_label = network(xyz14, overseg_idx, nodes, graph)
                loss, lcomp, lsmcomp, lseg = criterion(pred_xyz, pred_label, xyzl, true_label, global_epoch)

                loss_sum += loss
                loss_comp += lcomp
                loss_smcomp += lsmcomp
                loss_seg += lseg
            log_string('eval loss: %f' % (loss_sum / (float(num_batches))))
            log_string('eval loss cham: %f' % (loss_comp / num_batches))
            log_string('eval loss sem cham: %f' % (loss_smcomp / num_batches))
            log_string('eval loss seg: %f' % (loss_seg / num_batches))
            curr_loss = loss_sum / float(num_batches)
            curr_curr = loss_comp / float(num_batches)
            curr_smcham = loss_smcomp / float(num_batches)
            curr_seg = loss_seg / float(num_batches)
            if curr_curr <= best_comp and curr_smcham <= best_smcomp and curr_seg <= best_seg and global_epoch > 256:
                best_loss = curr_loss
                best_comp = curr_curr
                best_smcomp = curr_smcham
                best_seg = curr_seg
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'loss': best_loss,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best loss: %f' % best_loss)
            log_string('Best loss cham: %f' % best_comp)
            log_string('Best loss smcham: %f' % best_smcomp)
            log_string('Best loss seg: %f' % best_seg)
        global_epoch += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
