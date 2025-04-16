"""
Module: colmap_pipeline
Contains functions for performing COLMAP-based 3D reconstruction and refinement.
"""

import os
import shutil
import copy
import numpy as np
import logging
import pycolmap

from utils import _name_to_ind, make_folders, clean_paths
from metrics import compute_overlaps_in_rec, sort_cameras_by_filename, overlap_between_two_images

def extract_features(db_path, image_path, image_list):
    """
    Run COLMAP feature extraction on the provided images.
    
    Parameters:
        db_path (str): Path to the COLMAP database.
        image_path (str): Path to the image directory.
        image_list (list): List of image filenames.
    """
    if len(image_list) > 1:
        base_path = os.path.dirname(db_path)
        img_list_path = os.path.join(base_path, "image_list.txt")
        with open(img_list_path, "w") as f:
            f.writelines('\n'.join(image_list))
        feat_extracton_cmd = (
            "colmap feature_extractor "
            "--database_path " + db_path +
            " --image_path " + image_path +
            " --ImageReader.single_camera 1 " +
            " --image_list_path " + img_list_path +
            " --ImageReader.camera_model OPENCV " +
            " --SiftExtraction.max_num_features 5000 " +
            " --SiftExtraction.use_gpu " + str(True)
        )
    else:
        feat_extracton_cmd = (
            "colmap feature_extractor "
            "--database_path " + db_path +
            " --image_path " + image_path +
            " --ImageReader.single_camera 1 " +
            " --ImageReader.camera_model OPENCV " +
            " --SiftExtraction.max_num_features 5000 " +
            " --SiftExtraction.use_gpu " + str(True)
        )

    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

def feature_matching(db_path, sequential):
    """
    Run COLMAP feature matching using either sequential or exhaustive matching.
    
    Parameters:
        db_path (str): Path to the COLMAP database.
        sequential (bool): Flag to choose sequential matching.
    """
    mode = "sequential_matcher" if sequential else "exhaustive_matcher"
    feat_matching_cmd = (
        "colmap " + mode +
        " --database_path " + db_path +
        " --SiftMatching.use_gpu " + str(True)
    )
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

def reconstruct(source_path, db_path, image_path, output_path, image_list, existing_sparse='', clean=True, sequential=True, threshold=0.8):
    """
    Perform COLMAP incremental mapping to reconstruct a 3D scene.
    
    Parameters:
        source_path (str): Base directory.
        db_path (str): Path to the COLMAP database.
        image_path (str): Directory containing images.
        output_path (str): Directory to save the reconstruction.
        image_list (list): List of images to use.
        existing_sparse (str, optional): Path to an existing sparse reconstruction.
        clean (bool, optional): Flag to clean the output directory.
        sequential (bool, optional): Use sequential matching if True.
        threshold (float, optional): Minimum fraction of images to be registered.
    
    Returns:
        Reconstruction object if successful, else None.
    """
    extract_features(db_path, image_path, image_list)
    feature_matching(db_path, sequential)
    reconstructions = pycolmap.incremental_mapping(db_path, image_path, output_path, input_path=existing_sparse)

    if clean:
        rec_fold = os.listdir(output_path)
        [shutil.rmtree(os.path.join(output_path, tmp)) for tmp in rec_fold]

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
    """
    Iteratively attempt reconstruction by gradually increasing the number of frames.
    
    Parameters:
        source_path (str): Base directory.
        db_path (str): Path to the COLMAP database.
        image_path (str): Directory containing images.
        output_path (str): Directory for saving reconstruction.
        image_list (list): Initial list of images.
        fmw: FFmpegWrapper object.
        video_n (str): Video filename.
        max_iter (int, optional): Maximum iterations.
    
    Returns:
        Tuple (reconstruction, frames_list) from the successful iteration.
    """
    n_frames = int(fmw.duration)
    print(f" testing reconstruction with {n_frames} frames")
    frames_list = fmw.get_list_of_n_frames(n_frames)
    rec = reconstruct(source_path, db_path, image_path, output_path, frames_list)
    itr = 1

    while rec is None and itr < max_iter:
        clean_paths(source_path, video_n, db_path)
        make_folders(source_path)
        n_frames = int(n_frames * 1.1)
        print(f" testing reconstruction with {n_frames} frames")
        frames_list = fmw.get_list_of_n_frames(n_frames)
        rec = reconstruct(source_path, db_path, image_path, output_path, frames_list)
        itr += 1
    
    return rec, frames_list

def compute_best_new_images(rec, frames_list, fmw, min_overlap, new_image_budget):
    """
    Computes new candidate images to add between pairs of existing images
    where the overlap is below min_overlap.
    
    Parameters:
      rec                : the current reconstruction (COLMAP reconstruction)
      frames_list        : list of frame paths (unused in this implementation,
                           but may be kept for compatibility)
      fmw                : instance of FFmpegWrapper used to get frame paths by index
      min_overlap        : the required minimum overlap (an integer) per consecutive pair.
      new_image_budget   : total number of new images to add in this iteration.
    
    Returns:
      new_images_list    : a sorted list of new candidate image paths.
    """
    # Get sorted image IDs (keys) from the reconstruction.
    sorted_ids = sort_cameras_by_filename(rec)
    imgs = [rec.images[i] for i in sorted_ids]

    # Compute overlap values for each consecutive pair.
    pair_overlaps = []
    for i in range(len(imgs) - 1):
        ov = overlap_between_two_images(imgs[i], imgs[i+1])
        if ov is None: ov = 0
        pair_overlaps.append(ov)
    
    # Identify candidate pairs with overlap below the specified minimum.
    candidate_pairs = []  # Each item is a tuple (idx1, idx2) from the original frame numbering.
    deficiencies = []     # How much below min_overlap the current overlap is.
    
    for i, ov in enumerate(pair_overlaps):
        if ov < min_overlap:
            # Extract frame numbers from image names (e.g., "00000123.jpeg")
            img_name1 = imgs[i].name
            img_name2 = imgs[i+1].name
            idx1 = _name_to_ind(img_name1)
            idx2 = _name_to_ind(img_name2)
            # Only consider if there is room to add at least one new image.
            if idx2 - idx1 > 1:
                deficiency = min_overlap - ov
                deficiencies.append(deficiency)
                candidate_pairs.append((idx1, idx2))
    
    total_deficiency = sum(deficiencies)
    new_images_set = set()  # Use a set to avoid duplicates.
    # For each candidate gap, allocate a portion of the new_image_budget.
    for (pair, deficiency) in zip(candidate_pairs, deficiencies):
        (i1, i2) = pair
        if total_deficiency > 0:
            # Proportionally allocate the budget for this gap.
            pair_budget = int(round((deficiency / total_deficiency) * new_image_budget))
        else:
            pair_budget = 0
        
        # Cap the budget if the gap contains fewer available images.
        available_slots = i2 - i1 - 1
        pair_budget = min(pair_budget, available_slots)
        
        if pair_budget > 0:
            # Select candidate new images uniformly in this gap (exclude endpoints)
            indices = np.linspace(i1 + 1, i2 - 1, pair_budget)
            indices = [int(round(x)) for x in indices]
            for ind in indices:
                new_images_set.add(fmw.ind_to_frame_name(ind))
                
    new_images_list = sorted(new_images_set)
    return new_images_list

def incremental_reconstruction(source_path, db_path, image_path, output_path, image_list, fmw, rec, n_images,
                               quality_threshold_avg=100,  
                               min_overlap=25, new_image_budget=50):
    """
    Incrementally improves the reconstruction by adding new images in low-overlap gaps,
    with an early stopping mechanism based on average and minimum overlap quality.
    
    Parameters:
      source_path             : base directory for the reconstruction process.
      db_path, image_path, output_path : paths used in the reconstruction.
      image_list              : current list of image paths.
      fmw                     : an instance of FFmpegWrapper.
      rec                     : current reconstruction.
      n_images                : target number of registered images.
      quality_threshold_avg   : desired minimum average overlap.
      min_overlap             : minimum allowed overlap between consecutive images (for candidate selection).
      new_image_budget        : maximum number of new images to add per iteration.
    
    Returns:
      rec2                    : the updated reconstruction.
      new_images              : candidate images from the last iteration.
    """
    # Initially compute candidate new images.
    new_images = compute_best_new_images(rec, image_list, fmw, min_overlap, new_image_budget)
    rec2 = rec
    rec2.write_binary(output_path)

    while len(rec2.images) < n_images and len(new_images) > 0:
        new_images = compute_best_new_images(rec2, image_list, fmw, min_overlap, new_image_budget)
        print(f"Adding {len(new_images)} new images")
        image_list.extend(new_images)
        # Remove duplicates while preserving order.
        image_list = list(dict.fromkeys(image_list))
        
        tmp = reconstruct(source_path, db_path, image_path, output_path, image_list, output_path, clean=False)
        if tmp is None:
            print(len(image_list))
            print("New reconstruction failed ...")
            break

        rec2 = tmp
        rec2.write_binary(output_path)
        
        # Evaluate quality by computing overlap metrics.
        # Compute overlaps for all consecutive image pairs.
        sorted_ids = sort_cameras_by_filename(rec2)
        imgs = [rec2.images[i] for i in sorted_ids]
        overlaps = [overlap_between_two_images(imgs[i], imgs[i+1]) for i in range(len(imgs)-1)]
        
        if overlaps:
            avg_overlap = np.mean(overlaps)
            min_ovl = np.min(overlaps)
        else:
            avg_overlap = 0
            min_ovl = 0
        
        print(f"Current average overlap: {avg_overlap}, current minimum overlap: {min_ovl}")
        
        # Early stopping: if both quality metrics are met, break.
        if (quality_threshold_avg is not None and avg_overlap >= quality_threshold_avg) \
           and (min_overlap is not None and min_ovl >= min_overlap):
            print(f"Early stopping triggered. Average overlap {avg_overlap} exceeds threshold "
                  f"{quality_threshold_avg} and minimum overlap {min_ovl} exceeds threshold {min_overlap}.")
            break

    return rec2, new_images


def filter_rec(rec_orig, img_path):
    """
    Filter the reconstruction to remove points and images with poor quality.
    
    Parameters:
        rec_orig: Original reconstruction object.
        img_path (str): Path to the images used in reconstruction.
    
    Returns:
        Filtered reconstruction object.
    """
    rec = copy.deepcopy(rec_orig)
    pcd = rec.points3D
    ids = np.array(list(rec.point3D_ids()))

    pcd_o3d = np.array([pcd[p].xyz for p in pcd])
    # The conversion to an Open3D PointCloud is omitted here;
    # it can be performed externally if visualization is needed.
    pcd_col = np.array([pcd[p].color for p in pcd])
    n_views = np.array([pcd[p].track.length() for p in pcd])
    rep_error = np.array([pcd[p].error for p in pcd])

    thr_views = np.percentile(n_views, 10)
    thr_error = np.percentile(rep_error, 90)

    # Remove 3D points that do not meet quality criteria.
    to_remove = np.where((((n_views > thr_views) * (rep_error < thr_error)) * 1) == 0)[0]
    to_remove_ids = ids[to_remove]
    [rec.delete_point3D(i) for i in to_remove_ids]

    imgs = [rec.images[i] for i in rec.images]
    ids = np.array([i for i in rec.images])
    n_points2d = np.asarray([i.num_points2D() for i in imgs])
    n_points3d = np.asarray([i.num_points3D for i in imgs])
    ratio_2d3d = n_points3d / n_points2d

    thr_2dviews = np.percentile(n_points2d, 10)
    thr_ratio = np.percentile(ratio_2d3d, 10)

    # Remove images that do not meet the 2D and 3D point criteria.
    to_remove = np.where((((n_points2d > thr_views) * (ratio_2d3d > thr_ratio)) * 1) == 0)[0]
    to_remove_ids = ids[to_remove]
    [rec.deregister_image(i) for i in to_remove_ids]
    for i in to_remove_ids:
        p_to_rem = rec.images[i].get_observation_points2D()
        [rec.delete_point3D(i.point3D_id) for i in p_to_rem]
        os.remove(os.path.join(img_path, rec.images[i].name))
        del rec.images[i]

    print(f"Original images: {len(rec_orig.images)} -> Filtered images: {len(rec.images)}")
    print(f"Original 3D points: {len(rec_orig.points3D)} -> Filtered 3D points: {len(rec.points3D)}")
    return rec

def do_one(source_path, n_images, clean=False, minimal=False, average_overlap=100):
    """
    Main pipeline function to process a video, perform reconstruction, 
    and generate undistorted outputs.
    
    Parameters:
        source_path (str): Base directory containing video and images.
        n_images (int): Target number of images for reconstruction.
        clean (bool, optional): Flag to clean existing paths before processing.
    """
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
    if clean:
        clean_paths(source_path, video_n, db_path)
    make_folders(source_path)

    from video_processing import FFmpegWrapper
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
        rec2, frames_list = incremental_reconstruction(source_path, db_path, fmw.tmp_path, distorted_sparse_path, frames_list, fmw, rec, n_images, quality_threshold_avg=average_overlap)
                            
    print(rec2.summary())
    db_fin_path = os.path.join(distorted_path, "database_final.db")
    distorted_sparse_0_path = os.path.join(distorted_sparse_path, "0")
    os.makedirs(distorted_sparse_0_path, exist_ok=True)
    if os.path.isfile(os.path.join(sparse_path, "images.bin")):
        print("Loading final reconstruction")
        final = pycolmap.Reconstruction(sparse_path)
        frames_list = [os.path.join(input_p, final.images[i].name) for i in final.images]
    else:
        if minimal:
            frame_indices = select_minimal_image_subset(rec2, overlap_threshold=average_overlap)
        else:
            frame_indices = sorted([_name_to_ind(rec2.images[i].name) for i in rec2.images])
        fmw.extract_specific_frames(frame_indices)
        final = reconstruct(source_path, db_fin_path, input_p, distorted_sparse_final_path, sequential=False, image_list=[])
        final.write_binary(distorted_sparse_0_path)
    
    print(final.summary())

    if not minimal:
        from metrics import compute_overlaps_in_rec
        overl, _ = compute_overlaps_in_rec(rec)
        overl2, _ = compute_overlaps_in_rec(rec2)
        overlf, _ = compute_overlaps_in_rec(final)
        print(f"Original SFM: min_overlap: {np.min(overl)}; average_overl: {np.mean(overl)}; reconstruction summary: {rec.summary()}\n\n")
        print(f"Incremental SFM: min_overlap: {np.min(overl2)}; average_overl: {np.mean(overl2)}; reconstruction summary: {rec2.summary()}\n\n")
        print(f"Final SFM: min_overlap: {np.min(overlf)}; average_overl: {np.mean(overlf)}; reconstruction summary: {final.summary()}\n\n")

        print("filtering reconstruction....")
        final_filtered = filter_rec(final, input_p)
        print(final_filtered.summary())
        shutil.rmtree(distorted_sparse_0_path)
        os.makedirs(distorted_sparse_0_path, exist_ok=True)
        final_filtered.write_binary(distorted_sparse_0_path)


    img_undist_cmd = (
        "colmap image_undistorter "
        " --image_path " + input_p +
        " --input_path " + distorted_sparse_0_path +
        " --output_path " + source_path +
        " --output_type COLMAP"
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(source_path + "/sparse")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(source_path, "sparse", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def select_minimal_image_subset(rec, overlap_threshold=50):
    """
    From a reconstruction, select a minimal subset of image indices with sufficient overlap.

    Parameters:
        rec: pycolmap.Reconstruction object
        overlap_threshold (int): minimum required overlap between anchor and candidates

    Returns:
        List[int]: sorted list of selected frame indices
    """
    from metrics import sort_cameras_by_filename, overlap_between_two_images

    sorted_ids = sort_cameras_by_filename(rec)
    imgs = rec.images
    kept_ids = []
    i = 0
    while i < len(sorted_ids):
        anchor_id = sorted_ids[i]
        kept_ids.append(anchor_id)
        for j in range(i + 1, len(sorted_ids)):
            ov = overlap_between_two_images(imgs[anchor_id], imgs[sorted_ids[j]])
            if ov < overlap_threshold:
                i = j - 1
                break
        i += 1

    selected_indices = sorted([_name_to_ind(imgs[i].name) for i in kept_ids])
    print(f"[Minimal Subset] Selected {len(selected_indices)} frames out of {len(sorted_ids)}")
    return selected_indices
