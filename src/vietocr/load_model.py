import torch
import streamlit as st

from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


@st.cache(allow_output_mutation=True)
def load_model_vietocr():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~° ' + '̉'+ '̀' + '̃'+ '́'+ '̣'
    config['weights'] = 'weights/rec/VietOCR-best.pth'
    detector = Predictor(config)
    return detector