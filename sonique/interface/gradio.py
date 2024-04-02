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

from ..inference.generation import generate_diffusion_cond
from ..inference.priors import generate_mono_to_stereo
from ..stable_audio_tools.models.factory import create_model_from_config
from ..stable_audio_tools.models.pretrained import get_pretrained_model
from ..stable_audio_tools.models.utils import load_ckpt_state_dict
from ..inference.utils import prepare_audio
from ..stable_audio_tools.training.utils import copy_state_dict
from ..Video_LLaMA.inference import generate_prompt_from_video_description

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from moviepy.editor import VideoFileClip, AudioFileClip
import re


model = None
sample_rate = 32000
sample_size = 1920000

def add_music_to_video(video, music, output_path):
    v = VideoFileClip(video)
    m = AudioFileClip(music)
    m = m.subclip(0, min(m.duration, v.duration))
    demo_clip = v.set_audio(m)
    demo_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    v.close()
    m.close()
    return output_path

def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda"):
    global model, sample_rate, sample_size
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
        #model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        print(f"Done loading pretransform")

    model.to(device).eval().requires_grad_(False)

    print(f"Done loading model")

    return model, model_config

def generate_cond(
        # prompt,
        instruments,
        genres,
        tempo,
        negative_prompt=None,
        seconds_start=0,
        seconds_total=23,
        cfg_scale=6.0,
        steps=300,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-2m-sde",
        sigma_min=0.03,
        sigma_max=50,
        cfg_rescale=0.4,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        use_video=False,
        input_video=None,
        llms="mistral-7b",
        low_resource=True,
        mask_cropfrom=None,
        mask_pastefrom=None,
        mask_pasteto=None,
        mask_maskstart=None,
        mask_maskend=None,
        mask_softnessL=None,
        mask_softnessR=None,
        mask_marination=None,
        batch_size=1   
    ):
    import time
    start_time = time.time()

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None
    print(f'use video? {use_video}, use melody? {use_init}')
    prompt = f"{instruments}, {genres}, {tempo}"

    if prompt is not None:
        prompt = prompt.lower()

    print(prompt)
    # Return fake stereo audio
    conditioning = [{"prompt": prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}] * batch_size

    if negative_prompt:
        negative_conditioning = [{"prompt": negative_prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}] * batch_size
    else:
        negative_conditioning = None
        
    #Get the device from the model
    # device = next(model.parameters()).device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = int(seed) if int(seed) != -1 else np.random.randint(0, 2**31 - 1)
    if not use_video:
        input_video = None
        video_duration = 0
    # print(input_video)
    if input_video is not None:
        video_clip = VideoFileClip(input_video)
        video_duration = video_clip.duration
        if video_duration > 23:
            video_clip = video_clip.subclip(0, 23)
        video_des = generate_prompt_from_video_description(cfg_path="sonique/Video_LLaMA/eval_configs/video_llama_eval_only_vl.yaml", model_type="llama_v2", gpu_id="0", input_file=input_video, low_resource=low_resource)
        print(video_des)
        # Low resource code adapt from: https://huggingface.co/blog/4bit-transformers-bitsandbytes
        # Qwen
        if llms=="qwen-7b":
            if low_resource:
                llm = AutoModelForCausalLM.from_pretrained(
                        "Qwen/Qwen1.5-7B-Chat",
                        # "Qwen/Qwen1.5-14B-Chat",
                        quantization_config=BitsAndBytesConfig(
                                                                load_in_4bit=True,
                                                                bnb_4bit_quant_type="nf4",
                                                                bnb_4bit_use_double_quant=True,
                                                                bnb_4bit_compute_dtype=torch.float16
                                                                )
                    )
            else:
                llm = AutoModelForCausalLM.from_pretrained(
                        "Qwen/Qwen1.5-7B-Chat",
                        # "Qwen/Qwen1.5-14B-Chat",
                        device_map="cuda",
                        torch_dtype=torch.float16,
                    )
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen1.5-7B-Chat"
                # "Qwen/Qwen1.5-14B-Chat",
                )
            messages = [
                    {"role": "system", "content": "As a music composer fluent in English, you're tasked with creating background music for video. Based on the scene described, provide only one set of tags in English that describe this background music for the video. These tags must includes instruments, music genres, and tempo (BPM). Tags must be plain English words. Example of expected output: Piano, Synths, Strings, Violin, Flute, Reflective, Slow tempo, 96 BPM"},
                    {"role": "user", "content": str(video_des)}
                ]
            text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            llm_inputs = tokenizer([text], return_tensors="pt").to(llm.device)
            generated_ids = llm.generate(
                    llm_inputs.input_ids,
                    max_new_tokens=512
                )
            generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(llm_inputs.input_ids, generated_ids)
                ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        elif llms == "qwen-14b":
            if low_resource:
                llm = AutoModelForCausalLM.from_pretrained(
                        "Qwen/Qwen1.5-14B-Chat",
                        quantization_config=BitsAndBytesConfig(
                                                            load_in_4bit=True,
                                                            bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_use_double_quant=True,
                                                            bnb_4bit_compute_dtype=torch.float16
                                                            )
                    )
            else:
                llm = AutoModelForCausalLM.from_pretrained(
                        "Qwen/Qwen1.5-14B-Chat",
                        device_map="cuda",
                        torch_dtype=torch.float16,
                    )
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen1.5-14B-Chat"
                )
            messages = [
                    {"role": "system", "content": "As a music composer fluent in English, you're tasked with creating background music for video. Based on the scene described, provide only one set of tags in English that describe this background music for the video. These tags must includes instruments, music genres, and tempo (BPM). Avoid any non-English words. Example of expected output: Piano, Synths, Strings, Violin, Flute, Reflective, Slow tempo, 96 BPM"},
                    {"role": "user", "content": str(video_des)}
                ]
            text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            llm_inputs = tokenizer([text], return_tensors="pt").to(llm.device)
            generated_ids = llm.generate(
                    llm_inputs.input_ids,
                    max_new_tokens=512
                )
            generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(llm_inputs.input_ids, generated_ids)
                ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        elif llms == "mistral-7b":
            # Mistral - 7B
            if low_resource:
                llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", 
                                                        quantization_config=BitsAndBytesConfig(
                                                            load_in_4bit=True,
                                                            bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_use_double_quant=True,
                                                            bnb_4bit_compute_dtype=torch.float16
                                                            )
                                                        )
            else:
                llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", 
                                                        device_map="cuda", 
                                                        torch_dtype=torch.float16
                                                        )
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            messages = [{"role": "user", "content": f"As a music composer fluent in English, you're tasked with creating background music for video. \
                    Based on the scene described, provide only one set of tags in English that describe this background \
                    music for the video. These tags must include instruments, music genres, and tempo rate(e.g. 90 BPM). \
                    Avoid any non-English words. Please return the tags in the following JSON structure: {{\'tags\': [\'tag1\', \'tag2\', \'tag3\']}} \
                    Input: {video_des}"}]
            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
            llm_inputs = encodeds.to(llm.device)
            # llm.to(device)

            generated_ids = llm.generate(llm_inputs, max_new_tokens=1000, do_sample=True)
            decoded = tokenizer.batch_decode(generated_ids)
            print(decoded[0])
            responses = decoded[0]
            matched = re.findall(r"\{'tags': \[.*?\]\}", responses)
            if matched:
                json_str = matched[-1]

                json_str = json_str.replace("'", '"')

                try:
                    parsed_json = json.loads(json_str)

                    lst = parsed_json['tags']
                    response = ', '.join(lst)
                    print("Extracted Tags:", response)
                except json.JSONDecodeError as e:
                    print("Failed to parse JSON:", e)
            else:
                print("Failed to extract JSON string from response.")

        elif llms == "gemma-7b":    
            # Gemma - 7B
            if low_resource:
                llm = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", 
                                                           quantization_config=BitsAndBytesConfig(
                                                            load_in_4bit=True,
                                                            bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_use_double_quant=True,
                                                            bnb_4bit_compute_dtype=torch.float16
                                                            )
                                                                # device_map="auto", 
                                                                # torch_dtype=torch.float16,
                                                                # low_cpu_mem_usage=True,
                                                                # load_in_4bit=True
                                                                )
            else:
                llm = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", 
                                                                device_map="cuda", 
                                                                torch_dtype=torch.float16
                                                                )
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")

            inputs = f"As a music composer fluent in English, you're tasked with creating background music for video. \
                 Based on the scene described, provide only one set of tags in English that describe this background \
                 music for the video. These tags must include instruments, music genres, and tempo rate(e.g. 90 BPM). \
                 Avoid any non-English words. Please return the tags in the following JSON structure: {{\'tags\': [\'tag1\', \'tag2\', \'tag3\']}} \
                 Inputs: {video_des}"
            input_ids = tokenizer(inputs, return_tensors="pt").to(llm.device)
            outputs = llm.generate(**input_ids, max_new_tokens=512)
            responses = tokenizer.decode(outputs[0])
            responses = responses.split("Inputs:")[-1]
            print(responses)
            # Extract only tags from gemma response
            matched = re.findall(r"\{'tags': \[.*?\]\}", responses)
            if matched:
                json_str = matched[-1]

                json_str = json_str.replace("'", '"')

                try:
                    parsed_json = json.loads(json_str)

                    lst = parsed_json['tags']
                    response = ', '.join(lst)
                    print("Extracted Tags:", response)
                except json.JSONDecodeError as e:
                    print("Failed to parse JSON:", e)
            else:
                print("Failed to extract JSON string from response.")

        elif llms == "llama-7b":
            if low_resource:
                llm = AutoModelForCausalLM.from_pretrained(
                        "meta-llama/Llama-2-7b-chat-hf",
                        quantization_config=BitsAndBytesConfig(
                                                            load_in_4bit=True,
                                                            bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_use_double_quant=True,
                                                            bnb_4bit_compute_dtype=torch.float16
                                                            )
                    )
            else:
                llm = AutoModelForCausalLM.from_pretrained(
                        "meta-llama/Llama-2-7b-chat-hf",
                        device_map="cuda",
                        torch_dtype=torch.float16
                    )
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            messages = [
                                {"role": "system", "content": "As a music composer fluent in English, you're tasked with creating background music for video. \
                                Based on the scene described, provide only one set of tags in English that describe this background music for the video. \
                                These tags must includes instruments, music genres, and tempo (BPM). Avoid any non-English words. \
                                Example of expected output: Piano, Synths, Strings, Violin, Flute, Reflective, Slow tempo, 96 BPM \
                                Please return the tags in the following JSON structure: {{\'tags\': [\'tag1\', \'tag2\', \'tag3\']}}"},
                                {"role": "user", "content": str(video_des)}
                            ]
            text = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True
                            )
            llm_inputs = tokenizer([text], return_tensors="pt").to(llm.device)
            generated_ids = llm.generate(
                                llm_inputs.input_ids,
                                max_new_tokens=512
                            )
            generated_ids = [
                                output_ids[len(input_ids):] for input_ids, output_ids in zip(llm_inputs.input_ids, generated_ids)
                            ]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f'responses:{responses}')
            matched = re.findall(r"\{'tags': \[.*?\]\}", responses)
            if matched:
                json_str = matched[-1]

                json_str = json_str.replace("'", '"')

                try:
                    parsed_json = json.loads(json_str)

                    lst = parsed_json['tags']
                    response = ', '.join(lst)
                    print("Extracted Tags:", response)
                except json.JSONDecodeError as e:
                    print("Failed to parse JSON:", e)
            else:
                print("Failed to extract JSON string from response.")

        elif llms == "llama-13b":
            if low_resource:
                llm = AutoModelForCausalLM.from_pretrained(
                        "meta-llama/Llama-2-13b-chat-hf",
                        quantization_config=BitsAndBytesConfig(
                                                            load_in_4bit=True,
                                                            bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_use_double_quant=True,
                                                            bnb_4bit_compute_dtype=torch.float16
                                                            )
                    )
            else:
                llm = AutoModelForCausalLM.from_pretrained(
                        "meta-llama/Llama-2-13b-chat-hf",
                        device_map="cuda",
                        torch_dtype=torch.float16
                    )
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
            messages = [
                        {"role": "system", "content": "As a music composer fluent in English, you're tasked with creating background music for video. \
                         Based on the scene described, provide only one set of tags in English that describe this background music for the video. \
                         These tags must includes instruments, music genres, and tempo (BPM). Avoid any non-English words. \
                         Example of expected output: Piano, Synths, Strings, Violin, Flute, Reflective, Slow tempo, 96 BPM \
                         Please return the tags in the following JSON structure: {{\'tags\': [\'tag1\', \'tag2\', \'tag3\']}}"},
                         {"role": "user", "content": str(video_des)}
                            ]
            text = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True
                            )
            llm_inputs = tokenizer([text], return_tensors="pt").to(llm.device)
            generated_ids = llm.generate(
                                llm_inputs.input_ids,
                                max_new_tokens=512
                            )
            generated_ids = [
                                output_ids[len(input_ids):] for input_ids, output_ids in zip(llm_inputs.input_ids, generated_ids)
                            ]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(responses)
            matched = re.findall(r"\{'tags': \[.*?\]\}", responses)
            if matched:
                json_str = matched[-1]

                json_str = json_str.replace("'", '"')

                try:
                    parsed_json = json.loads(json_str)

                    lst = parsed_json['tags']
                    response = ', '.join(lst)
                    print("Extracted Tags:", response)
                except json.JSONDecodeError as e:
                    print("Failed to parse JSON:", e)
            else:
                print("Failed to extract JSON string from response.")
        # Clean up memory
        del llm
        gc.collect()
        torch.cuda.empty_cache()
     
        current_prompt = conditioning[0]['prompt']
        current_elements = current_prompt.split(', ')
        new_elements = response.split(', ')
                
        print(f'current element: {current_elements}')
        print(f'new elements: {new_elements}')

        current_bpm = next((element for element in current_elements if 'bpm' in element), None)
        new_bpm = next((element for element in new_elements if 'BPM' in element), None)
        
        if current_bpm:
            current_elements.remove(current_bpm)
        if new_bpm:
            new_elements.remove(new_bpm)
        
        updated_elements = set(current_elements)
        updated_elements.update(new_elements)
        
        bpm_to_include = current_bpm if current_bpm else new_bpm
        bpm_to_include = bpm_to_include
        updated_prompt = ', '.join(sorted(updated_elements)) + (', ' + bpm_to_include if bpm_to_include else '')
        conditioning[0]['prompt'] = updated_prompt.lower()
    print(f'updated conditioning prompt: {conditioning}')

    if not use_init:
        init_audio = None
    
    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        # Turn into torch tensor, converting from int16 to float32
        init_audio = torch.from_numpy(init_audio).float().div(32767)
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0) # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:

            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        print(f'getting callback info: {callback_info}')
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    # If inpainting, send mask args
    # This will definitely change in the future
    if mask_cropfrom is not None: 
        mask_args = {
            "cropfrom": mask_cropfrom,
            "pastefrom": mask_pastefrom,
            "pasteto": mask_pasteto,
            "maskstart": mask_maskstart,
            "maskend": mask_maskend,
            "softnessL": mask_softnessL,
            "softnessR": mask_softnessR,
            "marination": mask_marination,
        }
    else:
        mask_args = None 
    
    # Do the audio generation
    audio = generate_diffusion_cond(
        model, 
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        sample_size=input_sample_size,
        sample_rate=sample_rate,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        mask_args = mask_args,
        callback = progress_callback if preview_every is not None else None,
        scale_phi = cfg_rescale
    )

    # Convert to WAV file
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save("output.wav", audio, sample_rate)
    end = time.time()
    print(f'Total process time: {end - start_time}')
    
    # Let's look at a nice spectrogram too
    if use_video: 
        demo_video = add_music_to_video(input_video, "output.wav", "output.mp4")
        audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)
        return ("output.wav", demo_video, [audio_spectrogram, *preview_images], updated_prompt)

    else:
        audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)
        return ("output.wav", None, [audio_spectrogram, *preview_images], prompt)



def clear_all():
    return "", "", "", "", 0, 23, 7.0, 300, 0, -1, "dpmpp-2m-sde", 0.03, 80, 0.2, False, None, 3, False, None, "mistral-7b"

case_note_upload = ("""
### Some examples provided at the bottom of the page. Click on them to try them out!
""")

def create_sampling_ui(model_config, inpainting=False):

    model_conditioning_config = model_config["model"].get("conditioning", None)

    has_seconds_start = False
    has_seconds_total = False

    if model_conditioning_config is not None:
        for conditioning_config in model_conditioning_config["configs"]:
            if conditioning_config["id"] == "seconds_start":
                has_seconds_start = True
            if conditioning_config["id"] == "seconds_total":
                has_seconds_total = True
    with gr.Row():
        with gr.Column(scale=6):
            use_video = gr.Checkbox(label="Use video", value=False)
            video_input = gr.Video(label="Input video(23 secs max)")
            gr.Markdown(case_note_upload)
        with gr.Column(scale=6):
            instruments = gr.Textbox(label="Optional: enter instruments", placeholder="Enter desired instruments. E.G: piano, drums...")
            genres = gr.Textbox(label="Optional: enter genres", placeholder="Enter desired genres. E.G: rock, jazz...")
            tempo = gr.Textbox(label="Optional: enter tempo rate", placeholder="Enter desired tempo rate. E.G: 120 bpm,")
            negative_prompt = gr.Textbox(label="Optional: enter negative tags", placeholder="Negative tags - things you don't want in the output.")
            llms = gr.Dropdown(["mistral-7b", 
                                "gemma-7b", 
                                # "llama-7b", 
                                # "qwen-7b", 
                                "qwen-14b", 
                                "llama-13b"], 
                                label="LLMs", info="Select llm to extract video description to tags. Default Mistral-7B")
            low_resource = gr.Checkbox(label="Optional: To run the model in low_resource mode", value=True)
            generate_button = gr.Button("Generate", variant='primary', scale=1)
            clear_all_button = gr.Button("Clear all")

    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Accordion("Optional: use melody condition(inpaint)", open=False):
                with gr.Row():
                    init_audio_checkbox = gr.Checkbox(label="Use melody condition")
                    init_audio_input = gr.Audio(label="Melody condition audio")
                    init_noise_level_slider = gr.Slider(minimum=0.1, maximum=100.0, step=0.01, value=3, label="Init noise level")
            with gr.Accordion("Generation params", open=False):
                with gr.Row():
                    # Steps slider
                    steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=300, label="Steps")

                    # Preview Every slider
                    preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Preview Every")

                    # CFG scale 
                    cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=7.0, label="CFG scale")

                    seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Seconds start", visible=has_seconds_start)
                    seconds_total_slider = gr.Slider(minimum=0, maximum=512, step=1, value=sample_size//sample_rate, label="Seconds total", visible=has_seconds_total)
            with gr.Accordion("Sampler params", open=False):
            
                # Seed
                seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

                # Sampler params
                with gr.Row():
                    sampler_type_dropdown = gr.Dropdown(["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-2m-sde")
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=200.0, step=0.1, value=80, label="Sigma max")
                    cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.2, label="CFG rescale amount")
            
                inputs = [
                        # prompt, 
                        instruments,
                        genres,
                        tempo,
                        negative_prompt,
                        seconds_start_slider, 
                        seconds_total_slider, 
                        cfg_scale_slider, 
                        steps_slider, 
                        preview_every_slider, 
                        seed_textbox, 
                        sampler_type_dropdown, 
                        sigma_min_slider, 
                        sigma_max_slider,
                        cfg_rescale_slider,
                        init_audio_checkbox,
                        init_audio_input,
                        init_noise_level_slider,
                        use_video,
                        video_input,
                        llms,
                        low_resource
                    ]

    with gr.Row():
        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
        with gr.Column():
            video_output = gr.Video(label="Preview Video") 
            current_prompt = gr.Text(label="Currently used prompt")   

    generate_button.click(fn=generate_cond, 
        inputs=inputs,
        outputs=[
            audio_output, 
            video_output, 
            audio_spectrogram_output,
            current_prompt
        ], 
        api_name="generate")
    
    clear_all_button.click(fn=clear_all,inputs=[],outputs=[instruments,
                        genres,
                        tempo,
                        negative_prompt,
                        seconds_start_slider, 
                        seconds_total_slider, 
                        cfg_scale_slider, 
                        steps_slider, 
                        preview_every_slider, 
                        seed_textbox, 
                        sampler_type_dropdown, 
                        sigma_min_slider, 
                        sigma_max_slider,
                        cfg_rescale_slider,
                        init_audio_checkbox,
                        init_audio_input,
                        init_noise_level_slider,
                        use_video,
                        video_input,
                        llms])

    video_only_inputs = [
        use_video,
        video_input,
        init_audio_checkbox,
        init_audio_input,
        llms
    ]
    video_examples = gr.Examples(examples=[
        [True, "./demo_videos/Better_Call_Saul2.mp4", False, None, "mistral-7b"],
        [True, "./demo_videos/Better_Call_Saul4.mp4", False, None, "mistral-7b"],
        [True, "./demo_videos/breakingbad_4.mp4", False, None, "mistral-7b"],
        [True, "./demo_videos/breakingbad_6.mp4", False, None, "mistral-7b"],
        [True, "./demo_videos/friends_2.mp4", False, None, "mistral-7b"],
        ], 
                           inputs=video_only_inputs,
                           outputs=[audio_output, 
                                    video_output,
                                    audio_spectrogram_output,
                                    current_prompt], 
                                    fn=generate_cond,
                                    cache_examples=False,
                                    label="Example Video Input")
    video_with_melody = [
        init_audio_checkbox,
        init_audio_input,
        init_noise_level_slider,
        use_video,
        video_input,
        llms
    ]
    video_melody_examples = gr.Examples(examples=[
        [True,"./demo_videos/000590.wav", 3, True, "./demo_videos/Better_Call_Saul2.mp4", "mistral-7b"],
        [True,"./demo_videos/1908-1.wav", 3, True, "./demo_videos/breakingbad_6.mp4", "mistral-7b"],
        ], 
                           inputs=video_with_melody,
                           outputs=[audio_output, 
                                    video_output,
                                    audio_spectrogram_output,
                                    current_prompt], 
                                    fn=generate_cond,
                                    cache_examples=False,
                                    label="Example Video+Melody Input")
    
    prompt_input = [
        instruments,
        genres,
        tempo,
        init_audio_checkbox,
        init_audio_input,
        use_video,
        video_input
    ]
    prompt_examples = gr.Examples(examples=[
        ["Guitar, Drums, Bass", "Rock", "130 bpm", False, None, False, None],
        ["Piano", "Classical, Ambient, Slow", "80 bpm", False, None, False, None],
        ["Drums", "", "80 bpm", False, None, False, None]
        ], 
                           inputs=prompt_input,
                           outputs=[audio_output, 
                                    video_output,
                                    audio_spectrogram_output,
                                    current_prompt], 
                                    fn=generate_cond,
                                    cache_examples=False,
                                    label="Example Prompt Input")
    
    prompt_melody_input = [
        instruments,
        genres,
        tempo,
        init_audio_checkbox,
        init_audio_input,
        init_noise_level_slider,
        use_video,
        video_input
    ]
    prompt_melody_examples = gr.Examples(examples=[
        ["Guitar, Piano, Bass", "Jazz", "130 bpm", True, "./demo_videos/drums.wav", 5, False, None],
        ["Piano", "Ambient, Slow", "70 bpm", True, "./demo_videos/1908-4.wav", 3, False, None],
        ], 
                           inputs=prompt_melody_input,
                           outputs=[audio_output, 
                                    video_output,
                                    audio_spectrogram_output,
                                    current_prompt], 
                                    fn=generate_cond,
                                    cache_examples=False,
                                    label="Example Prompt+Melody Input")
    with gr.Blocks():

        with gr.Row():
            video_examples

        with gr.Row():
            video_melody_examples

        with gr.Row():
            prompt_examples
        with gr.Row():
            prompt_melody_examples
            
            



def create_txt2audio_ui(model_config):
    with gr.Blocks() as ui:
        gr.Markdown(
        """
        <h1 align="center">SONIQUE: Efficient Video Background Music Generation</h1>

        <h5 align="center">A Multi-model tool that designed to help video editors \
            generate background music on video & tv series' transition scene. \
            In addition, it can be used by music composers to generate conditioned \
            music base on instruments, genres, tempo rate, and even specific melodies. 
        </h5> 
        
        <h3>Video <span>&#8594;</span> Music:</h3>
        <li>1.Drop or upload videos to the section, check `use video` box.</li>
		<li>2.Optional: enter any desire instruments, genres and tempo on the right. \
            Also negative tags for things you don't want in the generated music. </li>
		<li>3.Choose a desire LLM from the list. Default: Mistral-7B.</li>
        <li>4.Click Generate button.</li>
        <li>Optional: You may upload melody as condition(inpaint) in below section.\
            You may also tune the Generation parameters and Sampler parameters.</li>

        To use without video, simply uncheck the `use video` box and enter any desired instruments, \
        genres, and tempo. Including any melody condition you want. 
        
        <Strong>Please note: Only uncheck `low_resource` mode if you have enough GPU memory (> 24GB) </strong>

        """
    )
        with gr.Tab("Generation"):
            create_sampling_ui(model_config) 
        # with gr.Tab("Inpainting"):
        #     create_sampling_ui(model_config, inpainting=True)    
    return ui

def diffusion_prior_process(audio, steps, sampler_type, sigma_min, sigma_max):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    audio = torch.from_numpy(audio).float().div(32767).to(device)
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0) # [1, n]
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1) # [n, 2] -> [2, n]

    audio = audio.unsqueeze(0)

    audio = generate_mono_to_stereo(model, audio, in_sr, steps, sampler_kwargs={"sampler_type": sampler_type, "sigma_min": sigma_min, "sigma_max": sigma_max})

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

def create_diffusion_prior_ui(model_config):
    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        # Sampler params
        with gr.Row():
            steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")
            sampler_type_dropdown = gr.Dropdown(["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-2m-sde")
            sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
            sigma_max_slider = gr.Slider(minimum=0.0, maximum=200.0, step=0.1, value=80, label="Sigma max")
        process_button = gr.Button("Process", variant='primary', scale=1)
        process_button.click(fn=diffusion_prior_process, inputs=[input_audio, steps_slider, sampler_type_dropdown, sigma_min_slider, sigma_max_slider], outputs=output_audio, api_name="process")    

    return ui

def create_ui(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None):

    assert (pretrained_name is not None) ^ (model_config_path is not None and ckpt_path is not None), "Must specify either pretrained name or provide a model config and checkpoint, but not both"

    if model_config_path is not None:
        # Load config from json file
        with open(model_config_path) as f:
            model_config = json.load(f)
    else:
        model_config = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model_config = load_model(model_config, ckpt_path, pretrained_name=pretrained_name, pretransform_ckpt_path=pretransform_ckpt_path, device=device)
    
    model_type = model_config["model_type"]

    if model_type == "diffusion_cond":
        ui = create_txt2audio_ui(model_config)
    elif model_type == "diffusion_prior":
        ui = create_diffusion_prior_ui(model_config)
        
    return ui