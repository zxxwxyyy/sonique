from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import time

"""
This script cleans up the tags to remove redundancies and prioritize important information using LLM (Qwen 14B).
"""


def prepare_llm_input(all_tags):
    combined_tags = ", ".join(all_tags)
    instruction = f"Given a detailed list of tags describing music tracks, \
                    your task is to simplify and prioritize the information. The tags include various attributes like instruments, genres, tempo, and mood descriptors. \
                    Please distill these tags into a concise set that captures the essence of the music. Focus primarily on the following categories: \
                    1. Instruments: Mention key instruments that define the track. 2. Genres: Identify the primary genre(s) the track belongs to. \
                    3. Tempo: Note the tempo if explicitly mentioned or implied by the tags. Limit the output to no more than 20 tags, \
                    ensuring to cover the most critical aspects of the track. Avoid redundancy by merging similar tags \
                    (e.g., combining 'Acoustic' and 'Piano' into 'Acoustic Piano' if applicable). The goal is to provide a distilled yet \
                    descriptive summary of the tags that can aid in categorizing and understanding the track's characteristics efficiently. \
                    Tags should be connected with comma. Example input tags: '4 on the floor, Acoustic, Ambient, Bass, Dance, Drumming, \
                    Energetic, Jazz cover, Piano, 123 BPM'. Expected output tags: 'Acoustic Piano, Ambient, Dance, Energetic, Jazz, 123 BPM'"
    
    return instruction, combined_tags

def generate_summarized_tags(instruction, combined_tags, model, tokenizer, device):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": combined_tags}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def process_and_save_metadata(file_path, output_path, model, tokenizer, device):
    with open(file_path, 'r') as file:
        metadata = json.load(file)
    
    new_metadata = {}
    for track, data in metadata.items():
        all_tags = set()
        for chunk in data.get('chunks', []):
            chunk_tags = chunk.get('tags', [])
            all_tags.update(chunk_tags)
        
        if all_tags:
            instruction, combined_tags = prepare_llm_input(all_tags)
            summarized_tags = generate_summarized_tags(instruction, combined_tags, model, tokenizer, device)
        else:
            summarized_tags = ""
       
        bpm = f'{data["bpm"]} BPM'
        new_metadata[track] = {
            "tags": summarized_tags + " ," + bpm
        }
    
        with open(output_path, 'w') as file:
            json.dump(new_metadata, file, indent=4)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-14B-Chat", device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat")

    start_time = time.time()
    process_and_save_metadata('./metadata_tags.json', './metadata_tags_cleaned.json', model, tokenizer, device)
    end_time = time.time()

    print(f"Processing completed in {end_time - start_time} seconds.")