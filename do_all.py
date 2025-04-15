import os
import shutil
from argparse import ArgumentParser
import time

def get_video_length(filename):
    import cv2
    video = cv2.VideoCapture(filename)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = int(frame_count / video_fps)

    return duration, frame_count, video_fps

def do_one(source_p, n_frames, clean=False, is_360=False,):

    start_time = time.time()

    files_n = os.listdir(source_p)
    video_n = None
    for f in files_n:
        if f.split(".")[-1] in ["mp4", "MP4"]:
            video_n = f
            break

    if video_n is None and (not ("input" in files_n)):
        exit(1)


    images_p = os.path.join(source_p, 'images')
    sparse_p = os.path.join(source_p, 'sparse')
    model_p = os.path.join(source_p, 'model')
    depths_p = os.path.join(source_p, 'd_images')


    if not (os.path.isdir(images_p) and os.path.isdir(sparse_p)):
        # extract frames, and perform SFM from video
        sfm_command = f"python preprocess/main_video_process.py -s {source_p} -n {n_frames}"
        if clean:
            sfm_command += " -c"
            
        print(sfm_command)
        exit_code = os.system(sfm_command)
        if exit_code != 0:
            print("error while performing sfm")
            exit(exit_code)

    sfm_time = time.time()

    # if is_360:
    #     if not "input" in files_n and not "img_360" in files_n:
    #         n_images_split = 16
    #         res = 2048
    #         n_images = len(os.listdir(input_p))
    #         tmp_path = os.path.join(source_p, "tmp")
    #         tmp_path_2 = os.path.join(source_p, "tmp2")
    #         path_360 = os.path.join(source_p, "img_360")
    #         shutil.move(input_p, path_360)
    #         os.makedirs(input_p, exist_ok=True)
    #         os.makedirs(tmp_path, exist_ok=True)
    #         os.makedirs(tmp_path_2, exist_ok=True)
    #         tmp_path_2 = os.path.join(tmp_path_2, "test.txt")
    #         split_exec = "/sugar/submodules/Meshroom-2023.3.0-linux/Meshroom-2023.3.0/aliceVision/bin/aliceVision_split360Images"
    #         split_imgs_comd = f"{split_exec} -i {path_360} -o {tmp_path} --equirectangularNbSplits {n_images_split} --equirectangularSplitResolution {res} --outSfMData {tmp_path_2} --nbThreads 24"
    #         exit_code = os.system(split_imgs_comd)
    #         if exit_code != 0:
    #             print("One error is expected when splitting images")
    #             # exit(exit_code)

    #         tmp_path_to_img = os.path.join(tmp_path, "rig")
    #         for i in range(n_images_split):
    #             in_p = os.path.join(tmp_path_to_img, f"{i}")

    #             for j in range(n_images):
    #                 in_im_p = os.path.join(in_p, "img_{:03d}.png".format(j + 1))
    #                 out_im_p = os.path.join(input_p, "img_{:03d}_{}.png".format(j+1, i))
    #                 shutil.move(in_im_p, out_im_p)


    if not os.path.isdir(depths_p):
        # estimate depth maps
        depth_command = f"cd /sugar/submodules/DepthAnythingV2_docker/ && python run.py --encoder vitl --pred-only --grayscale --img-path {images_p} --outdir  {depths_p}"
        exit_code = os.system(depth_command)
        if exit_code != 0:
            print("error while performing depth estimation")
            exit(exit_code)

    # estimating depth scale
    scale_cmd = f"conda run -n gaussian_splatting python /sugar/submodules/gaussian-splatting/utils/make_depth_scale.py --base_dir {source_p} --depths_dir {depths_p}"
    exit_code = os.system(scale_cmd)
    if exit_code != 0:
        print("error while performing depth scale estimation")
        exit(exit_code)

    depth_time = time.time()

    train_cmd = f"conda run -n gaussian_splatting python /sugar/submodules/gaussian-splatting/train.py " + \
        f"-s {source_p} -m {model_p} -d {depths_p} " + \
        "--exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp " + \
        "--data_device cpu --optimizer_type sparse_adam   --antialiasing"
    exit_code = os.system(train_cmd)
    if exit_code != 0:
        print("error while training")
        exit(exit_code)

    end_time = time.time()

    print(f"Total time: {end_time - start_time};\nsfm time: {sfm_time - start_time}\ndepth time: {depth_time - sfm_time}\ngs time: {end_time - depth_time}")


def main(args):

    source_p = args.source_path
    n_frames = args.max_number_of_frames
    clean = args.clean
    if not args.all:
        do_one(source_p, n_frames, clean)
    else:
        dirs = os.listdir(source_p)
        for d in dirs:
            tmp = os.path.join(source_p, d)
            if not os.path.isdir(tmp):
                continue
            do_one(tmp, n_frames, clean)



if __name__ == '__main__':
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--max_number_of_frames", "-n", default=400, type=int)
    parser.add_argument("--clean", "-c", action='store_true')
    parser.add_argument("--all", "-a", action='store_true')
    args = parser.parse_args()

    main(args)
