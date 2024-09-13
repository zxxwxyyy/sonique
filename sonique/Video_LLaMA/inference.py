import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from torch.cuda.amp import autocast

from sonique.Video_LLaMA.video_llama.common.config import Config
from sonique.Video_LLaMA.video_llama.common.dist_utils import get_rank
from sonique.Video_LLaMA.video_llama.common.registry import registry
from sonique.Video_LLaMA.video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
import gc

decord.bridge.set_bridge('torch')

from sonique.Video_LLaMA.video_llama.datasets.builders import *
from sonique.Video_LLaMA.video_llama.models import *
from sonique.Video_LLaMA.video_llama.processors import *
from sonique.Video_LLaMA.video_llama.runners import *
from sonique.Video_LLaMA.video_llama.tasks import *

decord.bridge.set_bridge('torch')
  

def generate_prompt_from_video_description(cfg_path, gpu_id, model_type, input_file, num_beams=1, temperature=1.0, low_resource=False):
    # initialize model
    args = argparse.Namespace(cfg_path=cfg_path, gpu_id=gpu_id, model_type=model_type, options=[])
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.low_resource = low_resource
    model_cls = registry.get_model_class(model_config.arch)

    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    if args.model_type == 'vicuna':
        chat_state = default_conversation.copy()
    else:
        chat_state = conv_llava_llama_2.copy()
    chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}')

    # process input
    if input_file.endswith('.jpg') or input_file.endswith('.png'):
        print(input_file)
        # chatbot = chatbot + [((input_file,), None)]
        chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        img_list = []
        llm_message = chat.upload_img(input_file, chat_state, img_list)
    elif input_file.endswith('.mp4'):
        print(input_file)
        # chatbot = chatbot + [((input_file,), None)]
        chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        img_list = []
        llm_message = chat.upload_video_without_audio(input_file, chat_state, img_list)

    else:
        print("Unsupported file type")
        return 
    
    question = "Describe the scene in detail"
    # question =  """
    #             As a music composer fluent in English, you're tasked with creating background music for a video. 
    #             Based on the scene described, provide a set of tags in English that describe this background music for the video. 
    #             Do not use the tags from the example. 
    #             Please only return the set of tags that describe this background music for the input video without any explanation.
    #             Return the tags in the following format: 
    #             Tags: [Tags1, Tags2, ..., Tempo (BPM)]
    #             Example format: 
    #             Tags: [Piano, Synths, Strings, Violin, Flute, Reflective, Slow tempo, 96 BPM]
    #             """
    with autocast():
        chat.ask(question, chat_state)

        llm_response = chat.answer(conv=chat_state,
                               img_list=img_list,
                               num_beams=num_beams,
                               temperature=temperature,
                               max_new_tokens=512,
                               max_length=2000)[0]
    print("Chatbot response:", llm_response)

    # clean up cache 
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return llm_response

