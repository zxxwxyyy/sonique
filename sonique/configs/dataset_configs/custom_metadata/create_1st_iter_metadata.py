import os
import json

class MetadataNotFoundError(Exception):
    pass

def get_custom_metadata(info, audio):
    if "slakh2100_1st_iter" in info['path']:
        metadata_file = '/scratch/lz2807/slakh2100_1st_iter/metadata_tags.json'
    elif "ballroom" in info['path']:
        metadata_file = '/scratch/lz2807/ballroom/audio/metadata_tags.json'
    elif "guitarset" in info['path']:
        metadata_file = '/scratch/lz2807/guitarset/metadata_tags.json'
    elif "medleydb_pitch" in info['path']:
        metadata_file = '/scratch/lz2807/medleydb_pitch/metadata_tags.json'
    elif "brid" in info['path']:
        metadata_file = '/scratch/lz2807/brid/metadata_tags.json'
    elif "candombe" in info['path']:
        metadata_file = '/scratch/lz2807/candombe/metadata_tags.json'
    elif "cuidado" in info['path']:
        metadata_file = '/scratch/lz2807/cuidado/metadata_tags.json'
    elif "gtzan_genres" in info['path']:
        metadata_file = '/scratch/lz2807/gtzan_genres/metadata_tags.json'
    elif "hainsworth" in info['path']:
        metadata_file = '/scratch/lz2807/hainsworth/metadata_tags.json'
    elif "maestro" in info['path']:
        metadata_file = '/scratch/lz2807/maestro/metadata_tags.json'
    elif "orchset" in info['path']:
        metadata_file = '/scratch/lz2807/orchset/metadata_tags.json'
    elif "smc-rock-simac-hjdb" in info['path']:
        metadata_file = '/scratch/lz2807/smc-rock-simac-hjdb/metadata_tags.json'
    elif "musicqa" in info['path']:
        metadata_file = '/scratch/lz2807/musicqa/metadata_tags.json'

    if not os.path.exists(metadata_file):
        raise MetadataNotFoundError(f"Metadata file not found in {metadata_file}")
    with open(metadata_file, 'r') as f:
            dataset_metadata = json.load(f)
    

    track_identifier = os.path.join(*info['path'].split('/')[-2:]) 

    if track_identifier not in dataset_metadata:
        raise MetadataNotFoundError(f"No metadata found for {track_identifier}")
    
    metadata = dataset_metadata[track_identifier]
    prompt = metadata
    print(f"Getting prompt {prompt} from track {track_identifier}")
    custom_metadata = {"prompt": prompt}
    print(custom_metadata)
    return custom_metadata