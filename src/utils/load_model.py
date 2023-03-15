import torch

# for streamlit
import streamlit as st

# for vietocr
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# for parseq
from strhub.data.module import SceneTextDataModule

# craft
import torch.backends.cudnn as cudnn
from src.craft.craft import CRAFT

# yolov8
from ultralytics import YOLO


# import handle code
from src.craft.craft_predict import copyStateDict
from src.craft.refinenet import RefineNet



# ==========
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





@st.cache(allow_output_mutation=True)
def load_model_vietocr():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~° ' + '̉'+ '̀' + '̃'+ '́'+ '̣'
    config['weights'] = 'weights/rec/VietOCR-best.pth'
    detector = Predictor(config)
    return detector

@st.cache(allow_output_mutation=True)
def load_model_parseq():
    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    return parseq, img_transform


@st.cache(allow_output_mutation=True)
def load_model_craft():
    net = CRAFT()     # initialize

    # choose cuda or cpu
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if refine:
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))

        refine_net.eval()

    return net, refine_net

@st.cache(allow_output_mutation=True)
def load_model_yolov8():
    model = YOLO(f'weights/detect/model_yolov8.pt')
    return model

