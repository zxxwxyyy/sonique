from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

start = time.time()
device = "cuda" 
torch.manual_seed(42)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-14B-Chat",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")

prompt = "The scene shows a city skyline at night, with the silhouette of the skyline at the beginning. It also shows the city from the water, with boats on the water. Throughout the video, the silhouette of the skyline remains, but there are changes in the lighting conditions."
messages = [
    {"role": "system", "content": "As a music composer fluent in English, you're tasked with creating background music for video. Based on the scene described, provide only one set of tags in English. These tags must focus on instruments, music genres, and tempo (BPM). Avoid any non-English words. Example of expected output: guitar, drums, bass, rock, 140 BPM"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
end = time.time()
print(end-start)