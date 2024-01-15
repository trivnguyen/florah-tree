
import sys
import os
import shutil
import subprocess
import glob
from pathlib import Path
from absl import flags
from ml_collections import config_dict, config_flags

from convert_scsam import convert_scsam
from prepare_scsam import prepare_scsam

def setup(config: config_dict.ConfigDict):

    # create all necessary directories
    workdir = Path(os.path.join(config.workdir, config.name))
    print('Creating work directory ...')
    if os.path.exists(workdir):
        print('Work directory already exists. Deleting ...')
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)

    # create an input and output directory
    input_dir = os.path.join(workdir, "input")
    output_dir = os.path.join(workdir, "output/")  # the slash is IMPORTANT. DO NOT REMOVE
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


if __name__ == "__main__":
    """ Setup and run SC-SAM and everything else """
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to SC-SAM configuration.",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    setup(FLAGS.config)
    convert_scsam(FLAGS.config)
    prepare_scsam(FLAGS.config)