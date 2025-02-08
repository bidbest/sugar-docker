import subprocess
import os
import sys
import json
from typing import List, Optional
import shutil
import pycolmap
from argparse import ArgumentParser
import copy

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

    def extract_specific_frames(self, frame_indices):
        # Convert frame indices to FFmpeg select filter format
        select_filter = "+".join([f"eq(n\\,{f})" for f in frame_indices])
        
        # Build FFmpeg command
        cmd = f"ffmpeg -i {self.video_path} -vf select='{select_filter}' -vsync vfr -pix_fmt rgb8 -q:v 4 {self.output_dir}/f_%08d.jpeg"

        exit_code = os.system(cmd)
        if exit_code != 0:
            print("error extracting frames")
            exit(exit_code)

    def ind_to_frame_name(self, ind):
            name = "f_{:08d}.jpeg".format(ind)
            return os.path.join(self.tmp_path, name)

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

def make_folders(source_path):

    input_p = os.path.join(source_path, 'input')
    distorted_path = os.path.join(source_path, "distorted")
    distorted_sparse_path = os.path.join(distorted_path, "sparse")
    distorted_sparse_final_path = os.path.join(distorted_path, "sparse_final")

    sparse_path = os.path.join(source_path, "sparse/0")

    os.makedirs(input_p, exist_ok=True)
    os.makedirs(distorted_path, exist_ok=True)
    os.makedirs(distorted_sparse_path, exist_ok=True)
    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(distorted_sparse_final_path, exist_ok=True)

def clean_paths(source_path, video_n, db_path):

    if os.path.isfile(db_path): os.remove(db_path)
    all_files = os.listdir(source_path)
    if video_n in all_files: all_files.remove(video_n)
    if "tmp" in all_files: all_files.remove("tmp")
    print(f"Clean start. removing:\n{all_files}")
    paths = [os.path.join(source_path, tmp) for tmp in all_files]
    [shutil.rmtree(tmp) for tmp in paths if os.path.isdir(tmp)]

def extract_features(db_path, image_path, image_list):

    if len(image_list) > 1:

        base_path = os.path.dirname(db_path)
        img_list_path = os.path.join(base_path, "image_list.txt")
        with open(img_list_path, "w") as f:
            f.writelines('\n'.join(image_list))

        feat_extracton_cmd = "colmap feature_extractor "\
            "--database_path " + db_path + "\
            --image_path " + image_path + "\
            --ImageReader.single_camera 1 " + "\
            --image_list_path " + img_list_path + "\
            --ImageReader.camera_model OPENCV " + "\
            --SiftExtraction.use_gpu " + str(True)

    else:
        feat_extracton_cmd = "colmap feature_extractor "\
            "--database_path " + db_path + "\
            --image_path " + image_path + "\
            --ImageReader.single_camera 1 " + "\
            --ImageReader.camera_model OPENCV " + "\
            --SiftExtraction.use_gpu " + str(True)

    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

def feature_matching(db_path, sequential):

    if sequential:
        mode = "sequential_matcher"
    else:
        mode = "exhaustive_matcher"
    
    ## Feature matching
    feat_matching_cmd = "colmap " + mode + "\
        --database_path " + db_path + "\
        --SiftMatching.use_gpu " + str(True)

    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)


def reconstruct(
    source_path, db_path, 
    image_path, output_path,
    image_list, existing_sparse = '',
    clean=True, sequential=True, threshold=0.8,
    ):
    
    # pycolmap.extract_features(db_path, image_path, 
    #     image_list=image_list, 
    #     camera_model="FULL_OPENCV", 
    #     camera_mode=pycolmap._core.CameraMode.SINGLE)

    extract_features(db_path, image_path, image_list)

    # if sequential:
    #     pycolmap.match_sequential(db_path)
    # else:
    #     pycolmap.match_exhaustive(db_path)
    feature_matching(db_path, sequential)


    reconstructions = pycolmap.incremental_mapping(db_path, image_path, output_path, input_path=existing_sparse)

    if clean:
        rec_fold = os.listdir(output_path)
        [shutil.rmtree(os.path.join(output_path, tmp)) for tmp in rec_fold]

    # If we don't have an image list, we want to use all images
    if len(image_list) == 0:
        image_list = os.listdir(image_path)

    min_registered_images = len(image_list) * threshold
    rec = None
    for tmp_rec in reconstructions:
        if reconstructions[tmp_rec].num_images() >= min_registered_images:
            rec = reconstructions[tmp_rec]
            break

    return rec


def iterative_reconstruc(source_path, db_path, image_path, output_path, image_list, fmw, video_n, max_iter=7):

    n_frames = int(fmw.duration)
    print(f" testing reconstruction with {n_frames} frames")
    frames_list = fmw.get_list_of_n_frames(n_frames)
    rec = reconstruct(source_path, db_path, image_path, output_path, frames_list)
    itr = 1

    while rec is None and itr < max_iter:

        clean_paths(source_path, video_n, db_path)
        make_folders(source_path)
        n_frames = int(n_frames*1.1)
        print(f" testing reconstruction with {n_frames} frames")
        frames_list = fmw.get_list_of_n_frames(n_frames)
        rec = reconstruct(source_path, db_path, image_path, output_path, frames_list)
        itr +=1
    
    return rec, frames_list


def _name_to_ind(name):
    tmp = name.split('.')[0]
    ind = int(tmp.split('f_')[-1])
    return ind


def compute_best_new_images(rec, frames_list, fmw):

    # displacements, peak_pairs_inds = compute_displacement_from_sorted(rec)
    
    overlap, peak_pairs_inds = compute_overlaps_in_rec(rec)

    peak_pairs = [(frames_list[p1-1].split("/")[-1], frames_list[p2-1].split("/")[-1]) for p1, p2 in peak_pairs_inds]
    new_images = []

    for p1, p2 in peak_pairs:

        i1, i2 = _name_to_ind(p1), _name_to_ind(p2)

        if i2 - i1 <=1:
            continue

        new_i = int((i1 + i2)/2)
        new_images.append(fmw.ind_to_frame_name(new_i))

    return new_images


def incremental_reconstruction(source_path, db_path, image_path, output_path, image_list, fmw, rec, n_images):

    new_images = compute_best_new_images(rec, image_list, fmw)
    rec2 = rec

    while len(rec2.images) < n_images and len(new_images) > 0:

        new_images = compute_best_new_images(rec2, image_list, fmw)
        print(f"Addin {len(new_images)} new images")
        image_list.extend(new_images)
        tmp = reconstruct(source_path, db_path, image_path, output_path, image_list, output_path, clean=False)

        if tmp is None:
            import pdb; pdb.set_trace()
            continue

        rec2 = tmp
        rec2.write_binary(output_path)

    return rec2, new_images


def overlap_between_two_images(image1, image2):
    
    a = set(image1.get_observation_point2D_idxs())
    b = set(image2.get_observation_point2D_idxs())

    return len(a.intersection(b))


def compute_overlaps_in_rec(rec):

    sorted_image_ids = sort_cameras_by_filename(rec)
    imgs = [rec.images[i] for i in rec.images]
    overl = [overlap_between_two_images(imgs[i], imgs[i+1]) for i in range(len(imgs)-1)]

    threshold = np.percentile(overl, 40)
    peak_pairs = [(sorted_image_ids[i], sorted_image_ids[i+1]) for i in range(len(overl)) if overl[i] <= threshold]

    return overl, peak_pairs


def filter_rec(rec_orig):

    rec = copy.deepcopy(rec_orig)   
    pcd = rec.points3D
    ids = np.array(list(rec.point3D_ids()))

    pcd_o3d = np.array([pcd[p].xyz for p in pcd])
    pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_o3d))
    pcd_col = np.array([pcd[p].color for p in pcd])

    n_views = np.array([pcd[p].track.length() for p in pcd])
    rep_error = np.array([pcd[p].error for p in pcd])

    thr_views = np.percentile(n_views, 60)
    thr_error = np.percentile(rep_error, 80)

    # Good views have: n_views > thr_views and  rep_error < thr_error. So all others are bad views.
    # by doing [True, False, False] * 1 you get [1, 0 ,0]; then chosing np.where == 0; is like doing not()
    to_remove = np.where((((n_views > thr_views) * (rep_error < thr_error)) * 1) == 0 )[0]
    to_remove_ids = ids[to_remove]

    [rec.delete_point3D(i) for i in to_remove_ids]

    pcd = rec.points3D
    ids = np.array(list(final.point3D_ids()))

    n_views = np.array([pcd[p].track.length() for p in pcd])


    



def do_one(source_path, n_images, clean=True):

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
    distorted_sparse_final_path = os.path.join(distorted_path, "sparse_final")
    sparse_path = os.path.join(source_path, "sparse/0")
    db_path = os.path.join(distorted_path, "database.db")
    if clean: clean_paths(source_path, video_n, db_path)
    make_folders(source_path)

    fmw = FFmpegWrapper(video_p, input_p)

    n_frames = int(fmw.duration)

    frames_list = fmw.get_list_of_n_frames(n_frames)

    if os.path.isfile(os.path.join(distorted_path, "orig_distorted", "images.bin")):
        print("Loading original reconstruction")
        rec = pycolmap.Reconstruction(os.path.join(distorted_path, "orig_distorted"))
        frames_list = [os.path.join(fmw.tmp_path, rec.images[i].name) for i in rec.images]
    else:
        rec, frames_list = iterative_reconstruc(source_path, db_path, fmw.tmp_path, distorted_sparse_path, frames_list, fmw, video_n)
        rec.write_binary(distorted_sparse_path)
        shutil.copytree(distorted_sparse_path, os.path.join(os.path.dirname(distorted_sparse_path), "orig_distorted"))
    
    print(rec.summary())


    if os.path.isfile(os.path.join(distorted_path, "sparse/0/", "images.bin")):
        print("Loading dense reconstruction")
        rec2 = pycolmap.Reconstruction(os.path.join(distorted_path, "sparse/0/"))
        frames_list = [os.path.join(fmw.tmp_path, rec2.images[i].name) for i in rec2.images]
    else:
        rec2, frames_list = incremental_reconstruction(source_path, db_path, fmw.tmp_path, distorted_sparse_path, frames_list, fmw, rec, n_images)
    
    print(rec2.summary())
    db_fin_path = os.path.join(distorted_path, "database_final.db")

    if os.path.isfile(os.path.join(sparse_path, "images.bin")):
        print("Loading final reconstruction")
        final = pycolmap.Reconstruction(sparse_path)
        frames_list = [os.path.join(input_p, final.images[i].name) for i in final.images]
    else:
        frame_indices = sorted([_name_to_ind(rec2.images[i].name) for i in rec2.images])
        fmw.extract_specific_frames(frame_indices)
        final = reconstruct(source_path, db_fin_path, input_p, distorted_sparse_final_path, image_list=[])

        distorted_sparse_0_path = os.path.join(distorted_sparse_path, "0")
        os.makedirs(distorted_sparse_0_path, exist_ok=True)
        final.write_binary(distorted_sparse_0_path)
    
    print(final.summary())

    img_undist_cmd = ("colmap image_undistorter \
        --image_path " + input_p + " \
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


    overl, _ = compute_overlaps_in_rec(rec)
    overl2, _ = compute_overlaps_in_rec(rec2)
    overlf, _ = compute_overlaps_in_rec(final)
    print(f"Original SFM: min_overlap: {np.min(overl)}; average_overl: {np.mean(overl)}; reconstruction summary: {rec.summary()}\n\n")
    print(f"Incremental SFM: min_overlap: {np.min(overl2)}; average_overl: {np.mean(overl2)}; reconstruction summary: {rec2.summary()}\n\n")
    print(f"Final SFM: min_overlap: {np.min(overlf)}; average_overl: {np.mean(overlf)}; reconstruction summary: {final.summary()}\n\n")

    
def main(args):

    source_path = args.source_path
    n_images = args.number_of_frames
    do_one(source_path, n_images)

if __name__ == '__main__':
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--number_of_frames", "-n", default=200, type=int)
    args = parser.parse_args()
    main(args)