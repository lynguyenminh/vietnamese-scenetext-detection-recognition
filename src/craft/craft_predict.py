
# support craft
from collections import OrderedDict
from src.craft import craft_utils, imgproc
import cv2
import torch
from torch.autograd import Variable

trained_model = './weights/detect/craft_mlt_25k.pth'
text_threshold = 0.7
low_text = 0.4
link_threshold = 0.4 
cuda = False
canvas_size = 1280
mag_ratio = 1.5
poly = False
show_time = False
test_folder = 'upload_image.jpg'
refine  = False
refiner_model = 'weights/craft_refiner_CTW1500.pth'


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, cuda, refine_net=None):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    return polys

def predict_craft(net, refine_net, image_path, text_threshold=0.65, cuda_state=False):
    image = imgproc.loadImage(image_path)
    polys = test_net(net, image, text_threshold=text_threshold, cuda=cuda_state, refine_net=refine_net)
    polys = [i.tolist() for i in polys]
    return polys
