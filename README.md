

# SONIQUE: Efficient Video Background Music Generation

<div style='display:flex; gap: 0.5rem; '>
<a href='https://zxxwxyyy.github.io/templates/sonique.html'><img src='https://img.shields.io/badge/Demo-Website-blueviolet'></a>
<!-- <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo(Coming Soon)-blue'></a> -->
<a href=''><img src='https://img.shields.io/badge/Thesis-Paper-pink'></a>
<a href='https://drive.google.com/file/d/1kRy-B82ZGvRrJq4M5ob45jOvQgp9r_Xz/view?usp=sharing'><img src='https://img.shields.io/badge/Dowload-Checkpoint-green'></a>
</div>

A Multi-model tool that designed to help video editors generate background music on video & tv series' transition scene. In addition, it can be used by music composers to generate conditioned music base on instruments, genres, tempo rate, and even specific melodies. Check out the [demo](https://zxxwxyyy.github.io/templates/sonique.html) page for more details. 

![t2i](demo_videos/assets/sonique.png)

**Performance:** Executing the entire process on an NVIDIA 4090 graphics card is accomplished in under a minute. This model requires less than 14 GB GPU memory. When operated on an NVIDIA 3070 Laptop GPU with 8 GB of memory, the process duration extends to 360 seconds.

# Table of contents
<!-- - [Demo](https://github.com/zxxwxyyy/sonique?tab=readme-ov-file#demo) -->
- [`Install`](#install)
- [`Model Checkpoint`](#model-checkpoint)
- [`Data Collection & Preprocessing`](#data-collection--preprocessing)
- [`Video-to-music-generation`](#video-to-music-generation)
- [`Text-to-music-generation`](#text-to-music-generation)
- [`Fine-tune LLM experiment`](#fine-tune-llm)
- [`Citation`](#citation)

# Install 
1. Clone this repo 
2. Create a conda environment: 
```bash
conda env create -f environment.yml
```
3. Activate the environment, navigate to the root, and run:
```bash
pip install .
```
4. After installation, you may run the demo with UI interface:
```bash
python run_gradio.py --model-config best_model.json --ckpt-path ./ckpts/stable_ep=220.ckpt
```
5. To run the demo without interface:
```bash
python inference.py --model-config best_model.json --ckpt-path ./ckpts/stable_ep=220.ckpt
```
### Additional inference flags:
- `--use-video`:
    - Use input video as condition
    - *Default*: False
- `--input-video`:
    - Path to input video 
    - *Default*: None
- `--use-init`:
    - Use melody condition
    - *Default*: False
- `init-audio`:
    - Melody condition path
    - *Default*: None
- `--llms`:
    - Selection of the name of Large Language Model to extract video description to tags
    - *Default*: Mistral 7B
- `--low-resource`:
    - If set to True, models from video -> tags stage will run in 4-bit. Only set it to False if you have enough GPU memory.
    - *Default*: True
- `--instruments`:
    - Input instrument condition
    - *Default*: None
- `--genres`:
    - Input genre condition
    - *Default*: None
- `--tempo-rate`:
    - Input tempo rate condition
    - *Default*: None
  
### Model Checkpoint
Pretrained model can be download [here](https://drive.google.com/file/d/1kRy-B82ZGvRrJq4M5ob45jOvQgp9r_Xz/view?usp=sharing). Please download, unzip, and save in the root of this project. 
```bash
sonique/
├── ckpts/
│   ├── .../
├── sonique/
├── run_gradio.py/
...
```

# Data Collection & Preprocessing

See [here](./data_preprocessing/README.md) for details.

# Video-to-music-generation
SONIQUE is a multi-model tool leveraging on [stable_audio_tools](https://github.com/Stability-AI/stable-audio-tools), [Video_LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA), and popular LLMs from Huggingface. 

Video description is extracted from the input video. I use [Video_LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA) to extract video description from the video. Then it will be pass to LLMs to converted them into tags that describe the background music. For the LLMs currently support: 
- Mistrial 7B (default)
- Qwen 14B
- LLaMA3 8B(You will need to get authenticate from [Meta](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct))
- LLaMA2 13B (You will need to get authenticate from [Meta](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf))
- Gemma 7B (You will need to get authenticate from [Google](https://huggingface.co/google/gemma-7b-it))

# Text-to-music-generation
Instead of using video, you may also mannually enter instruments, genres and tempo rate to generate music. You may upload melody as condition(inpaint) in `use melody condition` section. You may also tune the generation parameters and sampler parameters.

# Fine-tune LLM 
Fine-tuning the LLM at the `caption to tags` stage may improve the model's performance. To see that, I run an experiment fine-tuning `mistral-7b` with paired video description and audio tags. See [here](/fine_tune_llm/README.md) for the detailed process of how this is achieved. 

# Citation
Please consider citing the project if it helps your research:
```
@misc{zhang2024sonique,
  title={SONIQUE: Efficient Video Background Music Generation},
  author={Zhang, Liqian},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={https://github.com/zxxwxyyy/sonique},
}
```