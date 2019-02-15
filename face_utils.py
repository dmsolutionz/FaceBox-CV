import os, pathlib
import numpy as np
import torch

from data import cfg
from layers.functions.prior_box import PriorBox
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer
from utils.custom import nms_cpu, check_keys, load_model, remove_prefix


def load_faceboxes():
    """
    Load FaceBoxes model and weight in pytorch
    """

    pretrained_path = 'weights/FaceBoxes.pth'
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, pretrained_path)
    net.eval()
    print('Finished loading model')
    return net.cpu()



def get_facebox_coords(img, net, verbose=False):
    """
    Evaluates faceboxes forward pass and NMS
        returns dets, faceboxes array of [x1, x2, y1, y2, score]
    """
    # Initialize timer
    _t = {'forward_pass': Timer(), 'nms': Timer()}

    img = np.float32(img)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)

    # ========================================================== #
    # Forward pass evaluation                                    #
    # ========================================================== #

    _t['forward_pass'].tic()
    out = net(img)  # forward pass
    _t['forward_pass'].toc()
    _t['nms'].tic()

    priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
    priors = priorbox.forward()

    loc, conf, _ = out
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.data.cpu().numpy()[:, 1]


    # ========================================================== #
    confidence_threshold = 0.05
    top_k = 400
    # ========================================================== #

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K scores before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # Number of output predictions after thresholding
    if verbose:
        print('boxes after threshold:', len(scores))
        print('forward pass: {:.4f} s'.format(_t['forward_pass'].average_time))

    # ================= Note: NMS implemented ===================== #

    nms_threshold = 0.3
    keep_top_k = 10

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms_cpu(dets, nms_threshold)
    dets = dets[keep, :]

    # Trim results above top-K
    dets = dets[:keep_top_k]
    _t['nms'].toc()

    if len(dets.shape) == 1:  # edge case where nms has a single return
        dets = [dets]
    
    if verbose:
        print('Applying Non-Max Supression \nFace boxes:', len(dets))
        print('Forward pass and nms: {:.4f} s'.format(_t['nms'].average_time))
    return dets