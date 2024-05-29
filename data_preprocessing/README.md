

# Data collection & preprocessing for SONIQUE

![t2i](../demo_videos/assets/dataset.png)

To properly preprocess and create metadata for training, see the following steps:

## Remove silence part from instrument stems
The dataset used to train SONIQUE contains musical data collected without metadatas. [`Remove_silence.py`](remove_silence.py) removes the silent parts from isolated instrument stems to help the model learn more effectively.

## Scrape royalty-free background music
Part of the data used to train the conditioned music generation model was royalty-free music scraped from online. This ensures that the result is more tailored to video background music. [`Scrape_royalty_free_bgm.py`](scrape_royalty_free_bgm.py) uses the Python library `selenium` to automatically download royalty-free background music from Pixabay.

## Metadata creation for the training data 
[`Generate_audio_metadata.py`](generate_audio_metadata.py) leverages the pretrained model [`lp-music-caps`](https://github.com/seungheondoh/lp-music-caps) to generate metadata for the scraped music above. 

## Transfer captions to tags and Clean up tags
The output from lp-music-caps is split into 10-second segments. The initial output is long and challenging for the model to learn. I use two functions: [`convert_captions_to_tags.py`](convert_captions_to_tags.py) to first convert caption to tags; the use [`summarize_tags.py`](summarize_tags.py) to summarize them. These two function leverage on LLM (QWEN-14B) to clean them up. The final output looks like this:
```bash
"tags": "punchy bass, synth elements, female vocals, club atmosphere, emotional, bass, techno, 	energetic, danceable, 117 bpm"
```
