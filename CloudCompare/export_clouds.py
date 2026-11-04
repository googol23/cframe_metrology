#!/usr/bin/env python3
"""Extract standalone CloudComPy point clouds from a CloudCompare BIN project.

Usage: ./run.sh export_clouds.py /project/input/u3dl/data/scene.bin

Exports to the BIN's parent folder unless --output is given. The save operation
uses the CloudComPy ASCII writer, which determines included attributes.
"""
import argparse
import re
from pathlib import Path
import cloudComPy as cc


def clean_name(name):
    result = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', name).strip(' .')
    return result or 'UnnamedCloud'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path, help='CloudCompare .bin file')
    parser.add_argument('--output', type=Path, help='Destination directory (default: beside BIN)')
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing existing TXT files')
    args = parser.parse_args()
    project = args.project.expanduser().resolve(strict=True)
    outdir = (args.output or project.parent).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cc.initCC()
    result = cc.importFile(str(project))
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise RuntimeError(f'Unexpected importFile return: {type(result).__name__}')
    clouds = result[1]
    print(f'Project: {project}\nStandalone clouds: {len(clouds)}')
    used = set()
    for i, cloud in enumerate(clouds, 1):
        name = cloud.getName()
        base = clean_name(name)
        candidate = base
        suffix = 2
        while candidate.lower() in used or (not args.overwrite and (outdir / (candidate + '.txt')).exists()):
            candidate = f'{base}_{suffix}'
            suffix += 1
        used.add(candidate.lower())
        target = outdir / f'{candidate}.txt'
        print(f'[{i}/{len(clouds)}] {name!r}, {cloud.size()} points -> {target}')
        code = cc.SavePointCloud(cloud, str(target))
        if code != 0:
            raise RuntimeError(f'SavePointCloud failed ({code}) for {name!r}')
    print('Finished.')


if __name__ == '__main__':
    main()
