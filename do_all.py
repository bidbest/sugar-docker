import os
from argparse import ArgumentParser

def do_one(source_p, fps):

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
        extract_frames_cmd = f"ffmpeg -i {video_p} -r {fps} {input_p}/img_%03d.png"
        exit_code = os.system(extract_frames_cmd)
        if exit_code != 0:
            print("error extracting frames")
            exit(exit_code)

    if "images" not in files_n:
        print("performing SFM")
        sfm_cmd = f"python /sugar/submodules/gaussian-splatting-docker/convert.py -s {source_p}"
        exit_code = os.system(sfm_cmd)
        if exit_code != 0:
            print("error doing structure from motion frames")
            exit(exit_code)


    train_cmd = f"python /sugar/submodules/gaussian-splatting-docker/train.py -s {source_p} -r 4 -m {model_p} --data_device cpu"
    exit_code = os.system(train_cmd)
    if exit_code != 0:
        print("error whiletraining")
        exit(exit_code)


def main(args):

    source_p = args.source_path
    fps = args.frame_per_second
    if not args.all:
        do_one(source_p, fps)
    else:
        dirs = os.listdir(source_p)
        for d in dirs:
            tmp = os.path.join(source_p, d)
            if not os.path.isdir(tmp):
                continue
            do_one(tmp, fps)



if __name__ == '__main__':
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--frame_per_second", "-f", default=1, type=int)
    parser.add_argument("--all", "-a", action='store_true')
    args = parser.parse_args()

    main(args)
