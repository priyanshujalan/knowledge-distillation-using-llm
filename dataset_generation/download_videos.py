import json
import os

import yt_dlp

from common.config import BASE_DOWNLOAD_PATH, DOWNLOAD_LIMIT_PER_SOURCE
from common.file_helper import get_ffmpeg_version, get_path
from common.video_helpers import split_video



def download_shorts(dataset_name, urls, base_path=BASE_DOWNLOAD_PATH, download_limit=DOWNLOAD_LIMIT_PER_SOURCE, video_filter="duration < 61 & language ^= 'en'"):
    """
        Downloads shorts from a list of URLs into a specific folder.
    """
    download_path = os.path.join(base_path, dataset_name)
    print(f"\n🚀 STARTING CATEGORY: {dataset_name}")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Ensure MP4
        'outtmpl': f'{download_path}/%(id)s.%(ext)s',  # Filename: VideoID.mp4
        'match_filter': yt_dlp.utils.match_filter_func(video_filter),
        'playlistend': download_limit,  # Stop after X videos
        'js_runtimes': {'deno':  {'path': 'C:\ProgramData\chocolatey\lib\deno\deno.exe'}},
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',  # Ensures the final merged file is mp4
        }],
        'merge_output_format': 'mp4',

        'writedescription': True,  # Save description to .description file
        'writeinfojson': False,  # Don't Save tags/likes count to .json
        'writethumbnail': False,

        'quiet': False,
        'ignoreerrors': True  # Don't crash on private videos
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            print(f"--> Processing Source: {url}")
            try:
                ydl.download([url])
            except Exception as e:
                print(f"❌ Error downloading {url}: {e}")

    return download_path


def generate_dataset_from_json(json_file_name, root_path_for_split_videos):
    with open(json_file_name, "r") as f:
        data_sources = json.load(f)

    if not get_ffmpeg_version():
        print("FFMPEG version not available")
        return

    for category, meta_data in data_sources.items():
        print(f"Processing {category}")
        path_name = download_shorts(category, meta_data["links"], base_path=get_path(BASE_DOWNLOAD_PATH), download_limit=meta_data["download_limit"], video_filter=meta_data["filter_to_use"])
        # path_name = "Global_6000_Dataset\\Indian Videos"
        downloaded_mp4_files = [file_name for file_name in os.listdir(path_name) if file_name.endswith(".mp4")]

        # root_path_for_split_videos = "../dataset_videos"
        split_mp4_files = [file_name for file_name in os.listdir(root_path_for_split_videos) if file_name.endswith(".mp4")]
        split_mp4_files = list(set([f"{'-'.join(file_name.split('.mp4')[0].split('-')[0:-1])}.mp4" for file_name in split_mp4_files]))

        print(f"Processing {len(downloaded_mp4_files)} clips")
        for file_name in downloaded_mp4_files:
            if file_name in split_mp4_files:
                print(f"Skipping: {file_name}... Already Split")
                continue
            split_video(f"{path_name}\\{file_name}", root_path_for_output=root_path_for_split_videos)


if __name__ == "__main__":
    generate_dataset_from_json("../dataset_links.json")
