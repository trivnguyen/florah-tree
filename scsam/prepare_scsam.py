
import sys
import os
import shutil
import subprocess
import glob
from pathlib import Path
from absl import flags
from ml_collections import config_dict, config_flags

# DEFAULT
DEFAULT_PARAM_PATH = './default_params.scsam'

def replace_option(templates, header, values):
    ''' Replace the option in the template with the values '''
    for i in range(len(templates)):
        if templates[i].startswith(header):
            templates[i+1] = str(values) + '\n'
            break
    return templates

def prepare_scsam(config: config_dict.ConfigDict):
    ''' Run SC-SAM '''

    print('Preparing SC-SAM Parameters file ...')

    # create a work directory and copy the config file there
    workdir = Path(os.path.join(config.workdir, config.name))
    input_dir = os.path.join(workdir, "input")
    output_dir = os.path.join(workdir, "output/")  # the slash is IMPORTANT. DO NOT REMOVE

    # check how many files are there in the input directory and create a list of files
    input_files = glob.glob(os.path.join(input_dir, '*.dat'))
    input_files = sorted(input_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    files_list_path = workdir / 'files.list'
    with open(files_list_path, 'w') as f:
        f.write(f'{len(input_files)}\n')
        for i, file in enumerate(input_files):
            f.write(f'"{file}"\n')

    # Create parameter file for SC-SAM
    # read the default parameter file
    templates = []
    with open(DEFAULT_PARAM_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            templates.append(line)

    # replacing line with config
    # output directory
    templates = replace_option(
        templates, "#pathname of input and output", f'"{output_dir}"')

    # snapshots
    templates = replace_option(
        templates, "#NSNAP", config.num_snapshots)
    templates = replace_option(templates, "#NZOUT", len(config.min_snaps))
    str = ''
    for i in range(len(config.min_snaps)):
        str += f'{config.min_snaps[i]} {config.max_snaps[i]}'
        if i != len(config.min_snaps) - 1:
            str += '\n'
    templates = replace_option(templates, "#minsnap maxsnap", str)

    # write input files
    templates = replace_option(
        templates, "#filename of file containing list of tree filenames",
        f'"{files_list_path}"')

    # other options
    templates = replace_option(
        templates, "#usemainbranchonly", config.use_main_branch)

    # write a new template in work directory
    print('Creating parameter file for SC-SAM')
    with open(workdir / "params.scsam", 'w') as f:
        for line in templates:
            f.write(line)
            print(line, end='')
        f.write('\n')

    # Copy the binary file to the work directory and run it
    shutil.copy(config.binary_path, workdir)
    if config.run_scsam:
        print('Running SC-SAM ...')
        os.chdir(workdir)
        subprocess.run(['./gf', 'params.scsam'])
        print(f"SC-SAM run finished. Output directory: {output_dir}")

    print('Done!')


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to SC-SAM configuration.",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    prepare_scsam(config=FLAGS.config)
