import os
import numpy as np
import librosa
import soundfile as sf
import gc
import json

"""
This script removes silence part from separated stems in the dataset. 
"""

def fast_scandir(dir, ext):
    subfolders, files = [], []
    ext = ['.' + x if x[0] != '.' else x for x in ext]
    try:
        for f in os.scandir(dir):
            try:
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")
                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass
    except:
        pass
    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_progress(progress_file, progress):
    with open(progress_file, 'w') as f:
        json.dump(progress, f)


def remove_silence_from_files(directory, progress_file, extensions=['.flac'], chunk_duration=30.0, max_duration=600):
    _, files = fast_scandir(directory, extensions)
    progress = load_progress(progress_file)
    print(f'Found {len(files)} files to process.')

    for file in files:
        print(f"Processing {file}...")
        file_progress = progress.get(file, 0)
        total_duration = min(librosa.get_duration(filename=file), max_duration)
        sr = librosa.get_samplerate(file)

        all_chunks = []

        start = file_progress

        while start < total_duration:
            y, _ = librosa.load(file, sr=sr, offset=start, duration=chunk_duration)
            non_silent_intervals = librosa.effects.split(y, top_db=30)
            for interval_start, interval_end in non_silent_intervals:
                all_chunks.append(y[interval_start:interval_end])
            
            start += chunk_duration
            progress[file] = start  
            save_progress(progress_file, progress)

        if all_chunks:
            audio_without_silence = np.concatenate(all_chunks)
            sf.write(file, audio_without_silence, sr)
            print(f"Processed and overwritten: {file}")
        else:
            print(f"No audio with silence detected or file is too silent: {file}")

if __name__ == "__main__":
    directory = '...'
    progress_file = '...'

    remove_silence_from_files(directory, progress_file)