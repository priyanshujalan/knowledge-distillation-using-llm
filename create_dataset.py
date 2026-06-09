from common.config import CURRENT_TEACHER_MODEL
from common.file_helper import get_path
from dataset_generation.download_videos import generate_dataset_from_json
from dataset_generation.annotate_videos import annotate_downloaded_videos
from dataset_generation.generate_torch_dataset import get_torch_dataset

if input("Do you want to download Videos (Y/N)?") == "Y":
    generate_dataset_from_json(
        json_file_name=get_path("dataset_links.json"),
        root_path_for_split_videos=get_path("dataset_videos")
    )

if input("Do you want to Annotate Videos (Y/N)?") == "Y":
    annotate_downloaded_videos(
        videos_folder_path=get_path("dataset_videos"),
        db_file_path=get_path(f"checkpoints/datasets/json/{CURRENT_TEACHER_MODEL}_annotations.json"),
        csv_output=get_path(f"checkpoints/datasets/csv/{CURRENT_TEACHER_MODEL}_annotations.csv")
    )

if input("Do you want to create torch dataset (Y/N)?") == "Y":
    get_torch_dataset(
        file_path=get_path(f"checkpoints/datasets/csv/{CURRENT_TEACHER_MODEL}_annotations.csv"),
        output_file_prefix=f"{CURRENT_TEACHER_MODEL}_annotations"
    )

