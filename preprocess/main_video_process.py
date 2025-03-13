"""
Module: main
Entry point for running the COLMAP reconstruction pipeline.
"""

import argparse
from colmap_pipeline import do_one

def main(args):
    """
    Main function to parse arguments and run the pipeline.
    
    Parameters:
        args: Parsed command line arguments.
    """
    source_path = args.source_path
    n_images = args.number_of_frames
    clean = args.clean
    do_one(source_path, n_images, clean)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Colmap converter")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--number_of_frames", "-n", default=200, type=int)
    parser.add_argument("--clean", "-c", action='store_true')
    args = parser.parse_args()
    main(args)
