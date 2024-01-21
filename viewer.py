import os
from argparse import ArgumentParser

parser = ArgumentParser("Colmap converter")
parser.add_argument("--source_path", "-s", required=True, type=str)
args = parser.parse_args()

source_p = args.source_path
model_p = os.path.join(source_p, "model")

view_cmd = f"/sugar/submodules/gaussian-splatting-docker/SIBR_viewers/install/bin/SIBR_gaussianViewer_app -s {source_p} -m {model_p}"
exit_code = os.system(view_cmd)
if exit_code != 0:
    print("errorvisualizing")
    exit(exit_code)