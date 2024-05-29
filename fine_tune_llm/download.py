import os
import pandas as pd
from pytube import YouTube
from pytube.exceptions import LiveStreamError, VideoUnavailable, RegexMatchError
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip
import time
from http.client import IncompleteRead
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""
This script showcase how I download and trim the audio and videos collected in the csv. 
"""

def download_audio_video(youtube_url, output_path, retries=3):
    for attempt in range(retries):
        try:
            yt = YouTube(youtube_url)
            video_stream = yt.streams.filter(file_extension='mp4').first()
            audio_stream = yt.streams.filter(only_audio=True).first()
            
            if video_stream is None or audio_stream is None:
                raise VideoUnavailable("no suitable streams found")

            video_path = os.path.join(output_path, f"{yt.video_id}.mp4")
            audio_temp_path = os.path.join(output_path, f"{yt.video_id}.mp3")
            audio_wav_path = os.path.join(output_path, f"{yt.video_id}.wav")
            
            # avoid repeat downloading
            if os.path.exists(video_path) and os.path.exists(audio_wav_path):
                print(f"files already exist for {youtube_url}. skipping download.")
                return None, None

            video_stream.download(filename=video_path)
            audio_stream.download(filename=audio_temp_path)

            try:
                video_clip = VideoFileClip(video_path).subclip(0, 60)
                temp_video_path = os.path.join(output_path, f"{yt.video_id}_temp.mp4")
                video_clip.write_videofile(temp_video_path, codec='libx264')
                video_clip.reader.close()
                video_clip.audio.reader.close_proc()
                os.remove(video_path)
                os.rename(temp_video_path, video_path)
            except Exception as e:
                print(f"error trimming video for {youtube_url}: {e}")
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except PermissionError:
                        print(f"permission error when trying to remove {video_path}. skipping.")
                return None, None

            try:
                audio_clip = AudioFileClip(audio_temp_path).subclip(0, 60)
                audio_clip.write_audiofile(audio_wav_path)
                audio_clip.reader.close_proc()
                os.remove(audio_temp_path)
            except CouldntDecodeError as e:
                print(f"error decoding audio for {youtube_url}: {e}")
                if os.path.exists(audio_temp_path):
                    try:
                        os.remove(audio_temp_path)
                    except PermissionError:
                        print(f"permission error when trying to remove {audio_temp_path}. skipping.")
                return None, None

            return video_path, audio_wav_path

        except (LiveStreamError, VideoUnavailable, RegexMatchError, KeyError, AttributeError, IncompleteRead) as e:
            print(f"error downloading video {youtube_url}: {e}")
            if attempt < retries - 1:
                print(f"retrying... ({attempt + 1}/{retries})")
                time.sleep(5)
            else:
                print(f"failed to download {youtube_url} after {retries} attempts.")
                return None, None

def process_videos(input_csv, download_path, limit=10000, start_row=0):
    df = pd.read_csv(input_csv, encoding='utf-8')
    video_data = df[['Title', 'URL']].iloc[start_row:].to_dict('records')

    os.makedirs(download_path, exist_ok=True)

    downloaded_count = 0

    for video in video_data:
        if downloaded_count >= limit:
            break

        title = video['Title']
        url = video['URL']

        video_path, audio_path = download_audio_video(url, download_path)
        if video_path and audio_path:
            print(f"downloaded: {title}")
            print(f"video path: {video_path}")
            print(f"audio path: {audio_path}")
            downloaded_count += 1
        else:
            print(f"skipping: {title} ({url})")

if __name__ == "__main__":
    process_videos(input_csv="./cleaned_music_videos.csv", download_path="./mvs1")