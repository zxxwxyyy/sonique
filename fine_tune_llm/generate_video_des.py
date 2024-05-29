from sonique.Video_LLaMA.inference import generate_prompt_from_video_description
import os
import json
import torch
import gc

"""
This scripts shows how I create video description for downloaded YouTube MVs using Video_LLaMA. 
"""

def find_video_files(root_dir, video_extensions=['.mp4']):
    video_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    return video_files

def load_existing_metadata(dataset_root):
    json_file_path = os.path.join(dataset_root, 'video_metadata.json')
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)
    return {}

def find_missing_metadata_files(video_files, existing_metadata):
    missing_files = []
    for file in video_files:
        track_folder_name = os.path.basename(os.path.dirname(file))
        file_name = os.path.basename(file)
        track_identifier = f"{track_folder_name}/{file_name}"
        if track_identifier not in existing_metadata:
            missing_files.append(file)
    return missing_files
    
def captioning_batch(video_paths, dataset_root, existing_metadata, cfg_path, gpu_id, model_type):
    for video_path in video_paths:
        print(f'video path: {video_path}')
        try:
            description = generate_prompt_from_video_description(cfg_path, gpu_id, model_type, video_path)
            if description:
                track_folder_name = os.path.basename(os.path.dirname(video_path))
                file_name = os.path.basename(video_path)
                track_identifier = f"{track_folder_name}/{file_name}"
                existing_metadata[track_identifier] = {
                        "description": description
                    }
            json_file_path = os.path.join(dataset_root, 'video_metadata.json')
            with open(json_file_path, 'w') as json_file:
                json.dump(existing_metadata, json_file, indent=4)
        
        except Exception as e:
            print(f"Unexpected error processing {video_path}: {e}")
            continue
        gc.collect()
        torch.cuda.empty_cache()
    return existing_metadata

if __name__ == "__main__":
    existing_metadata = load_existing_metadata('./mvs1/')
    all_video_paths = find_video_files('./mvs1/')
    video_paths_to_process = find_missing_metadata_files(all_video_paths, existing_metadata)

    video_des = captioning_batch(video_paths_to_process, './mvs1/', existing_metadata, cfg_path="sonique/Video_LLaMA/eval_configs/video_llama_eval_only_vl.yaml", model_type="llama_v2", gpu_id="0")
    