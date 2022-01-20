from herbie.archive import Herbie
import datetime as dt
import sys
import io
import os
import json


def main():

    first = dt.datetime(2020, 1, 1, 12, 0)
    in_path = 'data'
    out_path = 'C:\\Users\\mit\\Dropbox (MIT)\\wildfire_repo'
    download_all_hrrr(first, 365, in_path, out_path, fxx=0)


def extract_hrrr(time, codes, save_dir, fxx=0):
    """ Downloads chosen subsets of HRRR file."""

    # get HRRR forecast initialization at this time (should be real-time weather)
    h = Herbie(time, model='hrrr', product='sfc', fxx=fxx, save_dir=save_dir)

    # generate regular expression for Herbie to download the right subsets
    herbie_regex = '|'.join(codes)

    # since Herbie does not tell us which subset is which other than through print statements
    # we have to use a hack to extract this information

    # start reading the stdout instead of printing
    original_out = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # download the desired data
    h.download(herbie_regex)

    # extract printed text to a variable, reset system to print to console
    output = new_stdout.getvalue()
    sys.stdout = original_out

    # get a lookup dictionary {code : bandNumber}
    subsets = output.split('GRIB_message')[1:]
    lookup = {code: [i for i, string in enumerate(subsets) if code in string] for code in codes}
    lens = list(set([len(lookup[key]) for key in lookup]))
    if lens != [1]:
        raise ValueError('Merge ambiguity for HRRR bands on download')
    band_lookup = {i: lookup[i][0] + 1 for i in lookup}

    return band_lookup


def download_all_hrrr(codes, first_dt, num_days, in_path, out_path, fxx=0):

    times = [(first_dt + dt.timedelta(days=i)) for i in range(num_days)]
    for time in times:
        direc = f'{in_path}/hrrr/{time.strftime("%Y%m%d")}'
        band_lookup = extract_hrrr(time, codes, 'data', fxx=0)
        hour = time.hour
        with open(f'{direc}/band_lookup_{hour}z_fxx{fxx}.json', 'w') as f:
            json.dump(band_lookup, f)

    # todo make this more like a copy, using os.walk
    os.rename(f'{in_path}/hrrr', f'{out_path}/hrrr')


if __name__ == '__main__':
    main()
