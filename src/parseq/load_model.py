import streamlit as st
import torch

from src.parseq.strhub.data.module import SceneTextDataModule
from src.parseq.strhub.models.utils import load_from_checkpoint


@st.cache(allow_output_mutation=True)
def load_model_parseq(device='cuda'):
    parseq = load_from_checkpoint('./weights/rec/best-parseq.ckpt').eval().to(device)
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    return parseq, img_transform

@st.cache(allow_output_mutation=True)
def load_model_parseq_author():
    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    return parseq, img_transform

