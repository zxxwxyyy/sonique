from moviepy.editor import VideoFileClip
import math

def split_video_into_chunks(input_video_path, chunk_length_sec=10):
    video = VideoFileClip(input_video_path)
    duration = video.duration  
    
    num_chunks = math.ceil(duration / chunk_length_sec)
    
    for i in range(num_chunks):
        start_time = i * chunk_length_sec
        end_time = min((i + 1) * chunk_length_sec, duration)
        
        chunk = video.subclip(start_time, end_time)
        
        output_filename = f"Better_Call_Saul{i+1}.mp4"
        
        chunk.write_videofile(output_filename, codec='libx264', audio_codec='aac')
        
        print(f"Chunk {i+1} saved as {output_filename}")
        
    video.close()

input_video_path = 'Better_Call_Saul.mp4'
split_video_into_chunks(input_video_path)