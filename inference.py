import argparse
import gc
import numpy as np
import gradio as gr
import json 
import torch
import torchaudio
import os
import random

from aeiou.viz import audio_spectrogram_image
from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T
from torch.cuda.amp import autocast

from efficient_video_bgm.inference.generation import generate_diffusion_cond
from efficient_video_bgm.inference.priors import generate_mono_to_stereo
from efficient_video_bgm.stable_audio_tools.models.factory import create_model_from_config
from efficient_video_bgm.stable_audio_tools.models.pretrained import get_pretrained_model
from efficient_video_bgm.stable_audio_tools.models.utils import load_ckpt_state_dict
from efficient_video_bgm.inference.utils import prepare_audio
from efficient_video_bgm.stable_audio_tools.training.utils import copy_state_dict
from efficient_video_bgm.Video_LLaMA.inference import generate_prompt_from_video_description
from efficient_video_bgm.interface.gradio import load_model, generate_cond

from transformers import AutoModelForCausalLM, AutoTokenizer
from moviepy.editor import VideoFileClip, AudioFileClip
import re

def main(args):
    torch.manual_seed(42)
    assert (args.pretrained_name is not None) ^ (args.model_config is not None and args.ckpt_path is not None), "Must specify either pretrained name or provide a model config and checkpoint, but not both"

    if args.model_config is not None:
        # Load config from json file
        with open(args.model_config) as f:
            model_config = json.load(f)
    else:
        model_config = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model(
        model_config=model_config,
        model_ckpt_path=args.ckpt_path,
        device=device
    )
    generate_cond(
        instruments=args.instruments,
        genres = args.genres,
        tempo=args.tempo_rate,
        use_init=args.use_init,
        init_audio=args.init_audio,
        use_video=args.use_video,
        input_video=args.input_video
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficient-video-bgm-generation script")
    parser.add_argument('--model-config', type=str, help='Path to model config', required=True)
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint', required=True)
    parser.add_argument('--pretrained-name', type=str, help='Optional:Name of pretrained model', required=False)
    
    parser.add_argument('--instruments', type=str, help='Optional:Input instruments condition', required=False, default="")
    parser.add_argument('--genres', type=str, help='Optional:Input genres condition', required=False, default="")
    parser.add_argument('--tempo-rate', type=str, help='Optional:Input tempo rate condition', required=False, default="")
    parser.add_argument('--use-init', type=bool, help='Optional:Use melody condition', required=False, default=False)
    parser.add_argument('--init-audio', type=str, help='Optional:Melody condition path', required=False, default=None)
    parser.add_argument('--use-video', type=bool, help='Optional:Use input video condition', required=False, default=False)
    parser.add_argument('--input-video', type=str, help='Optional:video condition path', required=False, default=None)

    args = parser.parse_args()
    
    main(args)    