import subprocess
import os
import json
from typing import List, Optional
import shutil
import pycolmap

class FFmpegWrapper:
    def __init__(self, video_path: str, output_dir: str):
        self.video_path = video_path
        self.output_dir = output_dir
        self.tmp_path = os.path.join(os.path.dirname(video_path), "tmp")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.tmp_path, exist_ok=True)
        self.fps, self.duration = self._get_video_metadata()
        self._extract_all_small_frames()
        self._get_frames_ids()
    
    def _get_video_metadata(self):
        """Retrieve FPS and duration of the video using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,duration",
            "-of", "json", self.video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        frame_rate = eval(metadata['streams'][0]['r_frame_rate'])
        duration = float(metadata['streams'][0]['duration'])
        return frame_rate, duration

    def _extract_all_small_frames(self):

        if len(os.listdir(self.tmp_path)) == 0:

            extract_frames_cmd = f"ffmpeg -i {self.video_path}  -pix_fmt rgb8 -q:v 4  -vf 'scale=-1:480' {self.tmp_path}/f_%08d.jpeg"
            exit_code = os.system(extract_frames_cmd)
            if exit_code != 0:
                print("error extracting frames")
                exit(exit_code)
        else:
            print("Frames already extracted")

    def _get_frames_ids(self):

        self.frames = sorted(os.listdir(self.tmp_path))

    def get_list_of_n_frames(self, n: int, start_frame: Optional[str] = None, end_frame: Optional[str] = None) -> List[str]:
        """Returns a list of n elements (equally distributed in time) without duplicates, up to the total available frames."""
        if not self.frames or n <= 0:
            return []
        
        # Determine valid range
        start_idx = self.frames.index(start_frame) if start_frame in self.frames else 0
        end_idx = self.frames.index(end_frame) if end_frame in self.frames else len(self.frames) - 1
        
        valid_frames = self.frames[start_idx:end_idx+1]
        total_frames = len(valid_frames)
        
        if n >= total_frames:
            print(f"Selected range has only {total_frames} frames. Returning all.")
            return [os.path.join(self.tmp_path, frame) for frame in valid_frames]
        
        step = total_frames / n
        indices = sorted(set(round(i * step) for i in range(n)))  # Ensure unique indices
        selected_frames = [valid_frames[i] for i in indices if i < total_frames]
        return [os.path.join(self.tmp_path, frame) for frame in selected_frames]
    
    def get_frames_between_pairs(self, peak_pairs: List[tuple], n: int) -> List[str]:
        """Returns a list of n images between each pair in peak_pairs using get_list_of_n_frames."""
        selected_frames = []
        for p1, p2 in peak_pairs:
            if p1 in self.frames and p2 in self.frames:
                selected_frames.extend(self.get_list_of_n_frames(n, start_frame=p1, end_frame=p2))
        return selected_frames

import numpy as np

def sort_cameras_by_filename(reconstruction):
    # Extract image names and sort by filename
    sorted_images = sorted(reconstruction.images.items(), key=lambda x: x[1].name)

    # Extract ordered image IDs
    sorted_image_ids = [img_id for img_id, _ in sorted_images]
    
    return sorted_image_ids

def identify_peaks(displacements, image_names, percentile=80):
    """Identifies peaks in displacement data and returns image name pairs where peaks occur."""
    threshold = np.percentile(displacements, percentile)
    peak_pairs = [(image_names[i], image_names[i+1]) for i in range(len(displacements)) if displacements[i] >= threshold]
    return peak_pairs

def compute_displacement_from_sorted(reconstruction):
    """
    Computes displacement between consecutive camera positions after sorting by filename.
    """
    sorted_image_ids = sort_cameras_by_filename(reconstruction)
    
    # Extract positions in sorted order
    positions = [reconstruction.images[i].cam_from_world.translation for i in sorted_image_ids]

    # Compute Euclidean displacement between consecutive frames
    displacements = [np.linalg.norm(positions[i+1] - positions[i]) for i in range(len(positions)-1)]

    tmp_peak_pairs = identify_peaks(displacements, sorted_image_ids, percentile=80)
    
    return displacements, tmp_peak_pairs


def clean_paths(source_path, video_n):

    all_files = os.listdir(source_path)
    if video_n in all_files: all_files.remove(video_n)
    if "tmp" in all_files: all_files.remove("tmp")
    print(f"Clean start. removing:\n{all_files}")
    paths = [os.path.join(source_path, tmp) for tmp in all_files]
    [shutil.rmtree(tmp) for tmp in paths if os.path.isdir(tmp)]


def do_one(source_path, reduction=4, clean=False):

    files_n = os.listdir(source_path)
    video_n = None
    for f in files_n:
        if f.split(".")[-1] in ["mp4", "MP4"]:
            video_n = f
            break

    if video_n is None and (not ("input" in files_n)):
        exit(1)

    video_p = os.path.join(source_path, video_n)
    input_p = os.path.join(source_path, 'input')
    distorted_path = os.path.join(source_path, "distorted")
    distorted_sparse_path = os.path.join(distorted_path, "sparse")
    sparse_path = os.path.join(source_path, "sparse/0")

    if clean: clean_paths(source_path, video_n)

    os.makedirs(distorted_path, exist_ok=True)
    os.makedirs(distorted_sparse_path, exist_ok=True)
    os.makedirs(sparse_path, exist_ok=True)

    fmw = FFmpegWrapper(video_p, input_p)

    db_path = os.path.join(distorted_path, "database.db")

    n_frames = int(fmw.duration)

    frames_list = fmw.get_list_of_n_frames(n_frames)

    pycolmap.extract_features(db_path, fmw.tmp_path, 
        image_list=frames_list, 
        camera_model="FULL_OPENCV", 
        camera_mode=pycolmap._core.CameraMode.SINGLE)

    pycolmap.match_exhaustive(db_path)
    # pycolmap.match_sequential(db_path)
    reconstructions = pycolmap.incremental_mapping(db_path, fmw.tmp_path, distorted_sparse_path)

    # Check if reconstructions are "good" or need improvements


    # remove reconstructions since automatically saved
    rec_fold = os.listdir(distorted_sparse_path)
    [shutil.rmtree(os.path.join(distorted_sparse_path, tmp)) for tmp in rec_fold]

    rec = None
    for tmp_rec in reconstructions:
        if reconstructions[tmp_rec].num_images() == n_frames:
            rec = reconstructions[tmp_rec]
            break

    displacements, peak_pairs_inds = compute_displacement_from_sorted(rec)
    peak_pairs = [(frames_list[p1].split("/")[-1], frames_list[p2].split("/")[-1]) for p1, p2 in peak_pairs_inds]
    frames_list.extend(fmw.get_frames_between_pairs(peak_pairs, 5))

    import pdb; pdb.set_trace()

    clean_paths(source_path, video_n)
    pycolmap.extract_features(db_path, fmw.tmp_path, 
        image_list=frames_list, 
        camera_model="FULL_OPENCV", 
        camera_mode=pycolmap._core.CameraMode.SINGLE)

    pycolmap.match_exhaustive(db_path)
    # pycolmap.match_sequential(db_path)
    reconstructions = pycolmap.incremental_mapping(db_path, fmw.tmp_path, distorted_sparse_path)

    distorted_sparse_0_path = os.path.join(distorted_sparse_path, "0")
    os.makedirs(distorted_sparse_0_path, exist_ok=True)
    rec.write_binary(distorted_sparse_0_path)

    img_undist_cmd = ("colmap image_undistorter \
        --image_path " + fmw.tmp_path + " \
        --input_path " + distorted_sparse_0_path +"\
        --output_path " + source_path + "\
        --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(source_path + "/sparse")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(source_path, "sparse", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)





source_path = "/sugar/datasets_gs/test/"
do_one(source_path, clean=True)