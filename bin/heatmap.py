import sys
import os
import argparse
import json
import cv2
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier, BACKBONES_TYPES# noqa
"""
python  heatmap.py /data/ly/experiments/csv/normal2500_bacterialTB2312.csv
 /home/yewenwu//classification/model/resnet34/resnet34_none_weigh/.cfg.json
/home/yewenwu//classification/model/resnet34/resnet34_none_weigh/best.ckpt
 /home/yewenwu//classification/model/resnet34/resnet34_none_weigh/
--num_workers 12 --device_ids '0,1,2,3'
"""
parser = argparse.ArgumentParser(description='Heat map')
parser.add_argument('pic_csv', default=None, metavar='PIC_CSV', type=str,
                    help="Path to the input image path in csv")
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in json format")
parser.add_argument('model_path', default=None, metavar='MODEL_PATH', type=str,
                    help="Path to a saved model e.g. ../../best.ckpt")
parser.add_argument('plot_path', default=None, metavar='PLOT_PATH', type=str,
                    help="Path to save plot images")
parser.add_argument('--heatmap_style', default='grid', type=str,
                    help="heatmap Style e.g. smooth or grid ")
parser.add_argument('--num_workers', default=1, type=int, help="Number of \
                    workers for each data loader")
parser.add_argument('--device_ids', default='0', type=str, help="GPU indices "
                    "comma separated, e.g. '0,1' ")
parser.add_argument('--show_logits', default=False, type=bool, help="If \
                     show logits on heatmap, default False")
parser.add_argument('--alpha', default=0.2, type=float, help="Transparancy \
                     alpha of the heatmap, default 0.2")

FONT_SIZE = {1024: 2, 768: 3, 512: 4, 256: 6, 224: 7}
MYFONT = matplotlib.font_manager.FontProperties(
    fname="/usr/share/fonts/simhei.ttf")


class SaveFeatures():
    features = None

    def __init__(self, m):
        torch.set_grad_enabled(False)
        m.eval()
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = (output)

    def remove(self): self.hook.remove()


def plot_pic(pic_son, cfg, args):
    np_pic = pic_son['feat_prob']
    np_logits = pic_son['feat_logits']
    plt.figure(figsize=(10, 5), dpi=300)
    ax = plt.subplot(1, 2, 2)
    print(pic_son['src_path'], '-->', pic_son['prob'])

    subtitle = 'avg logits: {:.4f} prob: {:.4f}'.format(
        np_logits.mean(),
        pic_son['prob'])
    plt.title(subtitle, fontproperties=MYFONT,
              fontsize=8, color='r')

    plt.imshow(pic_son['input_img'], cmap='gray', vmin=0, vmax=255)

    (H, W, C) = np_pic.shape

    np_pic_ = np.zeros((cfg.long_side, cfg.long_side))
    if args.heatmap_style == 'grid':
        # s_ means stride
        s_ = cfg.long_side // W

        for h in range(H):
            for w in range(W):
                np_pic_[h*s_:(h+1)*s_, w*s_:(w+1)*s_] = \
                    np_pic.squeeze(-1)[h, w]
                if args.show_logits:
                    lg = '{:.2f}'.format(np_logits.squeeze(-1)[h, w])
                    plt.text(w*s_ + s_/2, h*s_ + s_/2, lg,
                             fontsize=FONT_SIZE[cfg.long_side],
                             ha="center", va="center",
                             color='r')

    gci = plt.imshow(
        np_pic_ if args.heatmap_style == 'grid' else
        cv2.resize(np_pic, (cfg.long_side, cfg.long_side)),
        cmap='jet', vmin=0.0, vmax=1.0, alpha=args.alpha)
        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(gci, cax=cax)

    plt.subplot(1, 2, 1)
    plt.imshow(pic_son['input_img'], cmap='gray', vmin=0, vmax=255)

    label_str = ','.join(list(map(lambda x, y : '{}:{}'.format(x, y),
                                  pic_son['label_header'],
                                  pic_son['label'].astype(np.int16))))
    
    plt.suptitle(pic_son['src_path'] + '\n' + label_str,
                 fontproperties=MYFONT, fontsize=8, color='r')

    plt.savefig(pic_son['dst_path'], bbox_inches='tight')

    plt.close()


def get_logits(model, feature, cfg):
    torch.set_grad_enabled(False)
    output = []
    (N, C, H, W) = tuple(feature.shape)

    for index, num_task in enumerate(cfg.num_classes):
        classifier = getattr(model, "fc_" + str(index))
        out = torch.zeros((N, H, W, 1)).to(feature.device)
        for i in range(H):
            for j in range(W):
                out[:, i, j, :] = classifier(feature[:, :, i, j])
        output.append(out)
    return output


def gen_heatmap_epoch(args, cfg, dataloader, model, layer_name):
    torch.set_grad_enabled(False)
    model.eval()
    feature_layer = model.backbone._modules.get(layer_name)

    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    steps = len(dataloader)
    dataiter = iter(dataloader)

    # type(task_header) :list  e.g. ['TB', 'Normal']
    #iwith open(args.pic_csv) as f:
    #    test_header = f.readline().strip('\n').split(',')
    #    task_header = test_header[1:]
    
    test_header = ['Path', 'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural_Effusion']
    task_header = test_header[1:]

    num_tasks = len(cfg.num_classes)
    # assert len(task_header) == num_tasks, \
    #     'task number in csv file  does not equal to task number in cfg file'
    


    # mkdir for plot_path
    if not os.path.exists(args.plot_path):
        os.mkdir(args.plot_path)
    # mkdir for each task
    task_dir = []
    for task_name in task_header:
        task_dir.append(args.plot_path + '/' + task_name + '/')

        # mkdir for result pic
        if not os.path.exists(args.plot_path + '/' + task_name):
            os.mkdir(args.plot_path + '/' + task_name + '/')

    with open(args.plot_path + "/output.csv", 'w') as f:
        f.write(','.join(test_header) + '\n')
        for step in range(steps):
            activated_features = SaveFeatures(feature_layer)
            image, path, label = next(dataiter)
            image = image.to(device)
            batch_size = len(path)
            print("---------->step:", step, batch_size)

            # type(output):list [feature map task1, feature map of task 2 ,...]
            # shape(feature map task1 in output): BxHxW
            pred = model(image)
            features_x = activated_features.features
            # densenet
            if layer_name == 'features':
                features_x = F.relu(features_x)

            # FC layer
            output = get_logits(model, features_x, cfg)
            activated_features.remove()

            # shape(prob_feature): num_taskxBxHxW
            prob_feature = np.zeros((num_tasks,) + tuple(output[0].shape))
            logits_feature = np.zeros((num_tasks,) + tuple(output[0].shape))
            prob = np.zeros((num_tasks, batch_size))

            for t in range(num_tasks):
                prob_feature[t] = torch.sigmoid(
                    output[t]).cpu().detach().numpy()
                logits_feature[t] = output[t].cpu().detach().numpy()
                prob[t] = torch.sigmoid(
                    pred[t].view(-1)).cpu().detach().numpy()

            label = label.cpu().detach().numpy()
            image_ = image.cpu().detach().numpy()

            for i in range(batch_size):
                probs = ','.join(map(lambda x: '{:.5f}'.format(x), prob[:, i]))

                for t in range(num_tasks):
                    img_dst_name = '{:.4f}_{}'.\
                        format(prob[t, i], '_'.join(path[i].split('/')[-3:]))

                    pic_son = {'input_img': image_[i, 0, :, :]+cfg.pixel_mean,
                               'feat_prob': prob_feature[t, i, :, :],
                               'feat_logits': logits_feature[t, i, :, :],
                               'src_path': path[i],
                               'dst_path': task_dir[t] + img_dst_name,
                               'prob': prob[t, i],
                               'label': label[i, :],
                               'label_header': task_header}
                    plot_pic(pic_son, cfg, args)
                result = path[i] + ',' + probs
                f.write(result + '\n')


def run(args):
    with open(args.cfg_path) as f:
        cfg = edict(json.load(f))

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
            .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    # load model
    ckpt = torch.load(args.model_path, map_location=device)
    model = Classifier(cfg)
    model = model.to(device).eval()
    model.load_state_dict(ckpt['state_dict'])

    layer_name = ''
    if BACKBONES_TYPES[cfg.backbone] == 'resnet':
        layer_name = 'layer4'
    elif BACKBONES_TYPES[cfg.backbone] == 'densenet':
        layer_name = 'features'
    elif BACKBONES_TYPES[cfg.backbone] == 'inception':
        layer_name = 'Mixed_7c'
    else:
        raise Exception(
            'not support for : {} '
            .format(BACKBONES_TYPES[cfg.backbone]))

    # avoid the len of csv < cfg.batchsize
    heatmap_batchsize = cfg.dev_batch_size if cfg.dev_batch_size < (len(open(
        args.pic_csv).readlines()) - 1) else \
        len(open(args.pic_csv).readlines()) - 1

    # load data
    dataloader_heatmap = DataLoader(
        ImageDataset(args.pic_csv, cfg, mode='heatmap'),
        batch_size=heatmap_batchsize, num_workers=args.num_workers,
        drop_last=False, shuffle=False)
    gen_heatmap_epoch(args, cfg, dataloader_heatmap, model, layer_name)


def main():

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
