#!/usr/bin/env python

# usage: process_alc.py hosts/hostname/system_name/alchemical_name

import sys

from geometry import *

# dvdl.dat header line
# Step              Lambda                du/dl               Theta              vTheta                 Phi  ...


if __name__ == "__main__":

    full_path = sys.argv[1]
    ok, err_msg = check_geo_dir(full_path)
    if ok < 1:
        print(f"{full_path} is not a geometry directory")
        print(err_msg)
        sys.exit(1)
    dvdl = read_dvdls(full_path)
    dvdl.to_csv(f"{full_path}/dvdl.csv", index=False)
    dvdl.to_parquet(f"{full_path}/dvdl.parquet", index=False)

    fl = read_fls(full_path)
    fl.to_csv(f"{full_path}/fl.csv", index=False)
    fl.to_parquet(f"{full_path}/fl.parquet", index=False)
