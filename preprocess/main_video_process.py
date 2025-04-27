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
    full = args.full
    do_one(source_path, n_images, clean, minimal=args.minimal, full=full)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Colmap converter")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--number_of_frames", "-n", default=200, type=int)
    parser.add_argument("--clean", "-c", action='store_true')
    parser.add_argument("--minimal", "-m", action='store_true', help="Use minimal frame selection after final reconstruction")
    parser.add_argument("--full", "-f", action='store_true', help="Use all frame selection after final reconstruction")
    args = parser.parse_args()
    main(args)
