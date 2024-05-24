from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import time

"""
This script converts long captions into concise tags using LLM (Qwen 14B).
"""

def generate_tags_for_captions(captions, model, tokenizer, device):
    system_message = "Given a descriptive caption of a music track, your task is to read and analyze the description to identify the mood, genre, instruments, \
                      quality, and any specific themes or settings mentioned. Then, convert these elements into a concise list of tags that represent the essence of the music. \
                      The tags should be simple, keyword-based, and reflect genres, moods, settings, times, or any distinctive features mentioned. Avoid using long phrases or sentences; instead, \
                      focus on single words or short phrases that are commonly used as tags in music categorization (e.g., Trance, Ambient, Beach, High Energy, etc.). \
                      If the description mentions specific instruments, time of day, energy levels, or settings that imply a certain vibe or scene, include these as tags as well. \
                      Expected Output Tags: Instrumental, medium tempo, marching drum, cymbal, bass drum, percussive bass, amplified guitar, high quality, high adrenaline."
    
    tags = []
    for caption in captions:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": caption}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        tags.append(response)
    
    return tags

def process_and_save_metadata(file_path, output_path, model, tokenizer, device):
    with open(file_path, 'r') as file:
        metadata = json.load(file)
    
    new_metadata = {}
    for track, data in metadata.items():
        new_metadata[track] = {
            "bpm": data['bpm'],
            "chunks": [] 
        }
        for chunk in data['chunks']:
            caption = chunk['caption']
            raw_tags = generate_tags_for_captions([caption], model, tokenizer, device)
            tags = [tag.replace("Output Tags: ", "").strip() for tag in raw_tags]
            new_metadata[track]["chunks"].append({
                "start": chunk['start'],
                "end": chunk['end'],
                "tags": tags
            })
    
        with open(output_path, 'w') as file:
            json.dump(new_metadata, file, indent=4)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-14B-Chat", device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat")

    start_time = time.time()
    process_and_save_metadata('./dataset_metadata1.json', './metadata_tags.json', model, tokenizer, device)
    end_time = time.time()

    print(f"Processing completed in {end_time - start_time} seconds.")