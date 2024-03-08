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
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat")

prompt = "Acoustic guitar, Bass, Piano, Male vocals, Emotional, Passionate, Groove, Noisy, Commercial, Pop, Male vocals, Piano, Bass, Groove, Kick & snare, Hi hats, Shimmering, Addictive, Passionate, Pop, Male vocal, Electric guitar, Bass, Drums (kick, Snare, Hi hats), Piano, Synth pads, Upbeat, Cheerful, Acoustic guitar, Reverb, Delay, Fingerpicking, Dreamy, Bass loop, Electric bass, Live concert, Indie folk, Expected lowquality, Children'ssong, Malevocal, Piano, Bass, Groove, Bassdrum, Snare, Hihats, Passionate, Vocals (male), Piano, Bass, Snare, Hi-hats, Synth pads, Acoustic guitar, Emotional, Passionate, Noisy, Acoustic piano, Bass, Groove, Male vocal, Passionate, Shimmering hi hats, Snare drums, Acoustic, Piano, Bass, Hi-hats, Snare, Passionate, Emotional, Noisy, Acoustic guitar, Reverb, Delay, Fingerpicking, Dreamy, Ambient, Drum loop, Bass, Live concert, Indie folk, Vocal, Piano, Bass, Groove, Kick & snare, Hi hats, Happy, Fun, Joyful, Childrenssong, Malevocal, Bass, Kickdrum, Snare, Cymbals, Pianomelody, Electricguitar, Acousticguitar, Groovy, Passionate, Easygoing, Vocal, Piano, Bass, Shakers, Snare, Kick, Acoustic guitar, Emotional, Passionate, Noisy, Acoustic, Emotional, Passionate, Piano, Bass, Synth, Groove, Noisy, Commercial, Acoustic, Dreamy, Fingerpicked, Electric guitar, Reverb, Delay, Live, Drum loop, Bass, Jam session, Pop, Male vocals, Piano, Bass, Groove, Funky, Passionate, Bass drums, Snare, Electric guitar, Pop, Male vocal, Piano, Bass, Drumbeat, Hi hats, Upbeat, Happy, Fun, Acoustic, Electric guitar, Reverb, Delay, Fingerpicking, Dreamy, Drum loop, Bass, Live concert, Indie folk, Acoustic, Dreamy, Fingerpicking, Reverb, Delay, Live concert, Bass loop, Electric guitar, Bass, Acoustic, Piano, Soft, Mellow, Groovy, Bass, Rhythm guitar, Slow tempo"
messages = [
    {"role": "system", "content": "Given a detailed list of tags describing music tracks, your task is to simplify and prioritize the information. The tags include various attributes like instruments, genres, tempo, and mood descriptors. Please distill these tags into a concise set that captures the essence of the music. Focus primarily on the following categories: Instruments, Genres and Tempo rate. Limit the output to no more than 8 tags, ensuring to cover the most critical aspects of the track. Avoid redundancy by merging similar tags (e.g., combining 'Acoustic' and 'Piano' into 'Piano' if applicable). Example input tags: '4 on the floor, Acoustic, Ambient, Bass, Dance, Drumming, Energetic, Jazz cover, Piano, 123 BPM'. Expected output tags: 'Acoustic Piano, Ambient, Dance, Energetic, Jazz, 123 BPM. You should only return plain set of tags"},
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