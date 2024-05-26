import os
import argparse
import torch
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from model.bart import BartCaptionModel
from utils.audio_utils import load_audio, STR_CH_FIRST
import json
import librosa

"""
This script generates metadata for background music using the lp-music-caps model. 
The functions were created base on https://github.com/seungheondoh/lp-music-caps/blob/main/demo/app.py
"""

def download_pretrained_model():
    if not os.path.isfile("transfer.pth"):
        torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth', 'transfer.pth')
        torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/electronic.mp3', 'electronic.mp3')
        torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/orchestra.wav', 'orchestra.wav')


def load_model(device):
    model = BartCaptionModel(max_length=128)
    pretrained_object = torch.load('./transfer.pth', map_location='cpu')
    model.load_state_dict(pretrained_object['state_dict'])
    if torch.cuda.is_available():
        model = model.to(device)
    model.eval()
    return model

def get_bpm(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return round(tempo)

def find_audio_files(root_dir, audio_extensions=['.wav', '.mp3', '.flac']):
    audio_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files

def load_existing_metadata(dataset_root):
    json_file_path = os.path.join(dataset_root, 'dataset_metadata1.json')
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)
    return {}

def find_missing_metadata_files(audio_files, existing_metadata):
    missing_files = []
    for file in audio_files:
        track_folder_name = os.path.basename(os.path.dirname(file))
        file_name = os.path.basename(file)
        track_identifier = f"{track_folder_name}/{file_name}"
        if track_identifier not in existing_metadata:
            missing_files.append(file)
    return missing_files


def get_audio(audio_path, duration=10, target_sr=16000):
    n_samples = int(duration * target_sr)
    audio, sr = load_audio(
        path= audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= target_sr,
        downmix_to_mono= True,
    )

    if len(audio.shape) == 2:
        audio = audio.mean(0, False) 
    input_size = int(n_samples)
    if audio.shape[-1] < input_size: 
        pad = np.zeros(input_size)
        pad[: audio.shape[-1]] = audio
        audio = pad
    ceil = int(audio.shape[-1] // n_samples)
    audio = torch.from_numpy(np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32'))
    return audio

def captioning_batch(audio_paths, dataset_root, existing_metadata, model, device):
    for audio_path in audio_paths:
        audio_tensor = get_audio(audio_path = audio_path)
        if device is not None:
            audio_tensor = audio_tensor.to(device)
        with torch.no_grad():
            output = model.generate(
                samples=audio_tensor,
                num_beams=5,
            )
        chunks = []
        number_of_chunks = range(audio_tensor.shape[0])
        for chunk_index, text in zip(number_of_chunks, output):
            chunk_data = {
                "start": f"{chunk_index * 10}:00",
                "end": f"{(chunk_index + 1) * 10}:00",
                "caption": text
            }
            chunks.append(chunk_data)
        bpm = get_bpm(audio_path)
        track_folder_name = os.path.basename(os.path.dirname(audio_path))
        file_name = os.path.basename(audio_path)
        track_identifier = f"{track_folder_name}/{file_name}"
        existing_metadata[track_identifier] = {
            "bpm": bpm,
            "chunks": chunks
        }
    json_file_path = os.path.join(dataset_root, 'dataset_metadata1.json')
    with open(json_file_path, 'w') as json_file:
            json.dump(existing_metadata, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Batch process')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset containing audio files')
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    download_pretrained_model()
    model = load_model(device)
    existing_metadata = load_existing_metadata(args.dataset_root)
    all_audio_paths = find_audio_files(args.dataset_root)
    audio_paths_to_process = find_missing_metadata_files(all_audio_paths, existing_metadata)

    captioning_batch(audio_paths_to_process, args.dataset_root, existing_metadata, model, device)

if __name__ == "__main__":
    main()