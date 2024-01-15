
import os
import glob
import pandas as pd

def read_sam_table(
        datadir, min_snap, max_snap, prefix='galprop'):
    """ Read SAM data"""
    path = os.path.join(datadir, f'{prefix}_{min_snap}-{max_snap}.dat')
    print(f'read table from {path}')

    with open(path, 'r') as dat:
        lines   = dat.readlines()
        header  = [
            line.strip().split(" ")[2] for line in lines if line[0] == "#"]
    return pd.read_table(path, comment="#", delim_whitespace=True, names=header)

def read_sam_tables(datadir, prefix='galprop'):
    """ Read all SAM output in the directory """
    all_paths = glob.glob(os.path.join(datadir, f'{prefix}*'))
    all_tables = []
    for path in all_paths:
        with open(path, 'r') as dat:
            lines   = dat.readlines()
            header  = [
                line.strip().split(" ")[2] for line in lines if line[0] == "#"]
        table = pd.read_table(path, comment="#", delim_whitespace=True, names=header)
        all_tables.append(table)
    return pd.concat(all_tables)

def read_all_props(datadir):
    gal_table = read_sam_tables(datadir, prefix='galprop')
    halo_table = read_sam_tables(datadir, prefix='haloprop')
    return pd.merge(gal_table, halo_table, on='halo_index')