import pandas as pd
from youtubesearchpython import VideosSearch

"""
This script perform a search on YouTube for music videos, collect the url and title then save in a csv. 
"""

def get_music_video_data(query, total_videos, save_interval=30, filename="music_videos.csv"):
    all_video_data = []
    videos_search = VideosSearch(query, limit=30)
    saved_videos = 0

    while len(all_video_data) < total_videos:
        results = videos_search.result()['result']
        video_data = [{"Title": video['title'], "URL": video['link']} for video in results]
        all_video_data.extend(video_data)

        if len(all_video_data) - saved_videos >= save_interval:
            save_data_to_csv(all_video_data, filename, mode='a', header=(saved_videos == 0))
            saved_videos = len(all_video_data)

        # prevent infinity loop
        if len(all_video_data) >= total_videos:
            break
        
        # move to the next page of results
        videos_search.next()

    if saved_videos < len(all_video_data):
        save_data_to_csv(all_video_data, filename, mode='a', header=(saved_videos == 0))

    return all_video_data[:total_videos]

def save_data_to_csv(data, filename, mode='w', header=True):
    df = pd.DataFrame(data)
    df.to_csv(filename, mode=mode, header=header, index=False)

if __name__ == "__main__":
    query = "music video"
    total_videos = 50000 # a limit I use to control orevent infinity loop
    video_data = get_music_video_data(query, total_videos)