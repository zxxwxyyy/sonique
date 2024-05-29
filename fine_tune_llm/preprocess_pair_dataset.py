import json

"""
This script shows how I create paired datasets from video description and audio tags 
"""

with open('./video_metadata.json', 'r') as video_file:
    video_metadata = json.load(video_file)

with open('./metadata_tags.json', 'r') as audio_file:
    audio_metadata = json.load(audio_file)

paired_data = []
for video_id, video_data in video_metadata.items():
    audio_id = video_id.replace('.mp4', '.wav')
    
    if audio_id in audio_metadata:
        video_description = video_data['description']
        audio_tags = audio_metadata[audio_id]['tags']
        
        paired_data.append({
            "input": video_description,
            "output": audio_tags
        })

# system message I used to guide the llm, same message as use in SONIQUE
system_message = """As a music composer fluent in English, you're tasked with creating background music for video. \
                    Based on the scene described, provide only one set of tags in English that describe this background \
                    music for the video. These tags must include instruments, music genres, and tempo rate(e.g. 90 BPM). \
                    Avoid any non-English words. Please return the tags in the following JSON structure: {{\'tags\': [\'tag1\', \'tag2\', \'tag3\']}}"""

# convert paired data into conversational format
def create_conversation(entry):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": entry["input"]},
            {"role": "assistant", "content": entry["output"]}
        ]
    }

conversational_data = [create_conversation(entry) for entry in paired_data]

with open('train_dataset.json', 'w') as f:
    for item in conversational_data:
        f.write(json.dumps(item) + "\n")

# print the first 5 entries for verification
with open('train_dataset.json', 'r') as f:
    lines = f.readlines()
    for line in lines[:5]:
        print(json.loads(line))