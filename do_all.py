import os
import shutil
from argparse import ArgumentParser

def do_one(source_p, fps, is_360=False):

    files_n = os.listdir(source_p)
    video_n = None
    for f in files_n:
        if f.split(".")[-1] == "mp4":
            video_n = f
            break

    if video_n is None:
        exit(1)

    video_p = os.path.join(source_p, video_n)
    input_p = os.path.join(source_p, 'input')
    model_p = os.path.join(source_p, 'model')
    os.makedirs(input_p, exist_ok=True)

    if not "input" in files_n:
        print("extracting frames!")
        extract_frames_cmd = f"ffmpeg -i {video_p} -r {fps} -pix_fmt rgb8 {input_p}/img_%03d.png" # -pix_fmt rgb8 is needed for HDR 10bit videos
        exit_code = os.system(extract_frames_cmd)
        if exit_code != 0:
            print("error extracting frames")
            exit(exit_code)

    if is_360:
        if not "input" in files_n and not "img_360" in files_n:
            n_images_split = 16
            res = 2048
            n_images = len(os.listdir(input_p))
            tmp_path = os.path.join(source_p, "tmp")
            tmp_path_2 = os.path.join(source_p, "tmp2")
            path_360 = os.path.join(source_p, "img_360")
            shutil.move(input_p, path_360)
            os.makedirs(input_p, exist_ok=True)
            os.makedirs(tmp_path, exist_ok=True)
            os.makedirs(tmp_path_2, exist_ok=True)
            tmp_path_2 = os.path.join(tmp_path_2, "test.txt")
            split_exec = "/sugar/submodules/Meshroom-2023.3.0-linux/Meshroom-2023.3.0/aliceVision/bin/aliceVision_split360Images"
            split_imgs_comd = f"{split_exec} -i {path_360} -o {tmp_path} --equirectangularNbSplits {n_images_split} --equirectangularSplitResolution {res} --outSfMData {tmp_path_2} --nbThreads 24"
            exit_code = os.system(split_imgs_comd)
            if exit_code != 0:
                print("One error is expected when splitting images")
                # exit(exit_code)

            tmp_path_to_img = os.path.join(tmp_path, "rig")
            for i in range(n_images_split):
                in_p = os.path.join(tmp_path_to_img, f"{i}")

                for j in range(n_images):
                    in_im_p = os.path.join(in_p, "img_{:03d}.png".format(j + 1))
                    out_im_p = os.path.join(input_p, "img_{:03d}_{}.png".format(j+1, i))
                    shutil.move(in_im_p, out_im_p)



    if "images" not in files_n:
        print("performing SFM")
        sfm_cmd = f"python /sugar/submodules/gaussian-splatting-docker/convert.py -s {source_p}"
        exit_code = os.system(sfm_cmd)
        if exit_code != 0:
            print("error doing structure from motion frames")
            exit(exit_code)


    if is_360:
        reduction = 2
    else:
        reduction = 4

    train_cmd = f"python /sugar/submodules/gaussian-splatting-docker/train.py -s {source_p} -r {reduction} -m {model_p} --data_device cpu"
    exit_code = os.system(train_cmd)
    if exit_code != 0:
        print("error whiletraining")
        exit(exit_code)


def main(args):

    source_p = args.source_path
    fps = args.frame_per_second
    is_360 = args.is_360
    if not args.all:
        do_one(source_p, fps, is_360)
    else:
        dirs = os.listdir(source_p)
        for d in dirs:
            tmp = os.path.join(source_p, d)
            if not os.path.isdir(tmp):
                continue
            do_one(tmp, fps, is_360)



if __name__ == '__main__':
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--frame_per_second", "-f", default=1, type=int)
    parser.add_argument("--all", "-a", action='store_true')
    parser.add_argument("--is_360", "-i", action='store_true')
    args = parser.parse_args()

    main(args)
