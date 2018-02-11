# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import sys

sys.path.append("/home/fan/pytorch/pytorch-faster-rcnn/lib")
sys.path.append("/home/fan/pytorch/pytorch-faster-rcnn/tools")

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',), 'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def predict_bbox_of_images_to_file(images, output_file,
                                   allowed_classes=None, CONF_THRESH=0.8, NMS_THRESH=0.3):
    from tqdm import tqdm

    with open(output_file, 'w') as f:
        for im_path in tqdm(images):
            im = cv2.imread(im_path)
            scores, boxes = im_detect(net, im)

            item = {"img": im_path, "dets": {}}

            for cls_ind, cls in enumerate(CLASSES[1:]):
                if allowed_classes and cls not in allowed_classes:
                    continue

                cls_ind += 1  # because we skipped background
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(torch.from_numpy(dets), NMS_THRESH)
                dets = dets[keep.numpy(), :]
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

                if len(inds) == 0:
                    continue

                item["dets"][cls] = []
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    item["dets"][cls].append([list(bbox), score])

            print(item, file=f)


def draw_bbox_on_image(item, output_dir):
    font = cv2.FONT_HERSHEY_SIMPLEX

    im = cv2.imread(item["img"])
    dets = item["dets"]
    if len(dets.keys()) == 0:
        return

    for cls, bboxes in dets.items():
        for bbx, score in bboxes:
            bbx = map(int, bbx)
            cv2.rectangle(im, (bbx[0], bbx[1]), (bbx[2], bbx[3]), (255, 0, 0), 2)
    cv2.imwrite("%s/%s" % (output_dir, os.path.basename(item["img"])), im)

def crop_image_by_det(item, output_dir, cls="dog"):
    from PIL import Image
    im = Image.open(item["img"])
    dets = item["dets"]
    try:
        sorted_dets = sorted(dets[cls], key=lambda x: x[1], reverse=True)
        highest_score_bbx, score = sorted_dets[0]
        crop = im.crop(map(int, highest_score_bbx))
        crop.save("%s/%s" % (output_dir, os.path.basename(item["img"])))
        return 0
    except:
        print(item)
        return 1

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712 coco]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    saved_model = os.path.join('/home/fan/pytorch/pytorch-faster-rcnn/output', demonet, DATASETS[dataset][0],
                               NETS[demonet][0] % (70000 if dataset == 'pascal_voc' else 110000))

    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(21,
                            tag='default', anchor_scales=[8, 16, 32])

    net.load_state_dict(torch.load(saved_model))

    net.eval()
    net.cuda()

    print('Loaded network {:s}'.format(saved_model))

    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #             '001763.jpg', '004545.jpg']
    # for im_name in im_names:
    #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     print('Demo for data/demo/{}'.format(im_name))
    #     demo(net, im_name)
    #
    # plt.show()

    import glob
    images = glob.glob("/home/fan/dog-breed/*/*.jpg")
    print(len(images))

    predict_bbox_of_images_to_file(
        # map(lambda x: os.path.join("/home/fan/pytorch/pytorch-faster-rcnn/data/demo", x), im_names),
        images,
        output_file="/home/fan/dog-breed/out.txt",
        allowed_classes="dog",
        CONF_THRESH=0.5, NMS_THRESH=0.3
    )

    from tqdm import tqdm
    counter = 0
    for line in tqdm(open("/home/fan/dog-breed/out.txt").readlines()):
        item = eval(line)
        counter += crop_image_by_det(item, output_dir="/home/fan/tmp/dog-breed-crops")
        # draw_bbox_on_image(item, output_dir="/home/fan/tmp/dog-breed")
    print(counter)