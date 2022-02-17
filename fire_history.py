from osgeo import gdal
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopy.distance as dst
import os
import datetime as dt
import glob
from PIL import Image
import shutil

def preprocess(dropbox_path, scale=0.6, buffer_miles=0, gdal_path='data/prepared',
               clean_path='data/processed', gifs=False, weather=False):

    # prevent gdal from printing errors to command prompt without throwing error
    gdal.UseExceptions()

    # read in the fire observations
    boundaries = pd.read_pickle('data/clean/merged_boundaries.pkl')
    fire_pixels_lookup = pd.read_csv('data/clean/fire_pixels_lookup.csv')
    fire_pixels = pd.read_csv('data/clean/viirs.csv')

    # merge in the indident ids for the fire observations
    fire_pixels = pd.merge(fire_pixels, fire_pixels_lookup, on='pixel_id', how='right', validate='1:1')
    fire_pixels = fire_pixels[fire_pixels['INCIDENT_IDENTIFIER'].notna()]
    fire_pixels['INCIDENT_IDENTIFIER'] = fire_pixels['INCIDENT_IDENTIFIER'].astype(str)

    # read in the static data parameters (file locations)
    df = pd.read_csv('data/input/static_data_params.csv')
    df['path'] = dropbox_path + "\\" + df['path']

    # if out_path is not a directory, make it (needs directory one above to exist)
    if not os.path.isdir(gdal_path):
        os.mkdir(gdal_path)

    # for each fire
    for i, fire in boundaries.iterrows():

        # grab the bounding box and incident id
        bbox = fire['bbox']
        inc_id = fire['INCIDENT_IDENTIFIER']

        # define a subdirectory for this fire, create it if needed
        fire_out_path = f'{gdal_path}/{inc_id}'
        if not os.path.isdir(fire_out_path):
            os.mkdir(fire_out_path)

        # get the normalized bounding box and lon/lat resolution
        lon_res, lat_res, new_bbox = get_fire_grid_bounds(bbox, scale, buffer=buffer_miles)

        # get the static data for this bounding box
        ds = generate_static_data(df, lon_res, lat_res, new_bbox, fire_out_path)

        # get the daily fire footprint from this bounding box and the fire observation data
        _ = generate_daily_fire_footprint(inc_id, ds, fire_pixels, new_bbox, scale, gdal_path)

    # clean completely and write to clean path
    time_fxx = [{'time': '12z', 'fxx': 'f00'}]
    hrrr = dropbox_path + '/wildfire_repo/hrrr2'
    inc_ids = os.listdir(gdal_path)
    for inc_id in inc_ids:
        process_single_fire_data(gdal_path, clean_path, hrrr, inc_id, time_fxx)

    # no HRRR data outside of CONUS, drop here
    bad_weather = [i for i in os.listdir(clean_path) if
                   np.all(np.load(f'{clean_path}/{i}/1/weather_12z_f00.npy') == 0)]
    [shutil.rmtree(f'{clean_path}/{i}') for i in bad_weather]

    # make fire footprint gifs if needed
    if gifs:
        make_gifs(gdal_path)

    # get daily weather data if needed
    if weather:
        raise NotImplementedError()


def local_miles_per_lon_lat(bbox, k=100):
    """ Estimates miles per degree longitude, miles per degree latitude at the center of a bounding box."""

    # get lat, lon of center of bbox
    lat = (bbox[3] + bbox[1]) / 2
    lon = (bbox[2] + bbox[0]) / 2

    # get points very close to each other that vary a little bit in latitiude direction
    # note: dst.distance uses (lat, lon) convention
    coords1 = (lat - 1 / k, lon)
    coords2 = (lat + 1 / k, lon)

    # find miles per degree latitude
    miles_per_lat = dst.distance(coords1, coords2).miles * k / 2

    # get points very close to each other that vary a little bit in longitude direction
    coords1 = (lat, lon - 1 / k)
    coords2 = (lat, lon + 1 / k)

    # find miles per degree longitude
    miles_per_lon = dst.distance(coords1, coords2).miles * k / 2

    return miles_per_lon, miles_per_lat


def get_fire_grid_bounds(bbox, scale, buffer=0):

    # get the local miles per degree longitude, miles per degree latitude
    miles_per_lon, miles_per_lat = local_miles_per_lon_lat(bbox)

    # translate the scale in miles to a (lon, lat) resolution
    lon_res = scale / miles_per_lon
    lat_res = scale / miles_per_lat

    # extract the bounds to variables
    lon_min = bbox[0] - buffer / miles_per_lon
    lon_max = bbox[2] + buffer / miles_per_lon
    lat_min = bbox[1] - buffer / miles_per_lat
    lat_max = bbox[3] + buffer / miles_per_lat

    # find the length of the axes in the grid we will create
    lon_ticks = (lon_max - lon_min) / lon_res
    lat_ticks = (lat_max - lat_min) / lat_res

    # in case I want to add some buffer around the bbox
    if buffer != 0:
        extra_ticks = int(np.round(buffer / scale))
        lon_min - (buffer/miles_per_lon)
        lat_ticks += 2 * extra_ticks

    # buffer this grid to a whole number of ticks in each dimension
    lon_buffer = (1 - np.mod(lon_ticks, 1))
    if lon_buffer == 1:
        lon_buffer = 0
    lon_buffer = lon_buffer * lon_res
    lat_buffer = (1 - np.mod(lat_ticks, 1))
    if lat_buffer == 1:
        lat_buffer = 0
    lat_buffer = lat_buffer * lat_res
    new_bbox = [lon_min - lon_buffer / 2, lat_min - lat_buffer / 2,
                lon_max + lon_buffer / 2, lat_max + lat_buffer / 2]

    # return the resolutions and buffered bounding box
    return lon_res, lat_res, new_bbox


def generate_static_data(params_df, lon_res, lat_res, fire_grid_bbox, out_path):
    # initialize list of vrt files that will comprise output data set
    vrts = []

    # for each static parameter
    for _, row in params_df.iterrows():

        # get the path to the data set and open it
        path = row['path']
        ds = gdal.Open(path)

        # get the name of the layer, and define a path to a .vrt file we will write
        layer = row['layer']
        vrt_file = f'{out_path}/{layer}.vrt'

        # append to list of vrt files
        vrts.append(vrt_file)

        # remove .vrt file if it exists (maybe not needed, but GDAL misbehaves sometimes
        # and I think it might have to do with file overwriting)
        if os.path.isfile(vrt_file):
            os.remove(vrt_file)

        # write a vrt file to the specified location, which links to the original dataset,
        # clips to the specific region, and reshapes the data to the proper grid
        gdal.Warp(vrt_file, ds, format='VRT', dstSRS='WGS84', xRes=lon_res, yRes=lat_res,
                  resampleAlg=row['impute'], outputBounds=fire_grid_bbox)

    # define output file for all static data, delete if it exists
    out_file = f'{out_path}/static_data.vrt'
    if os.path.isfile(out_file):
        os.remove(out_file)

    # build a .vrt that references all the previous .vrts, one for each layer
    # note: we cannot move any of the referenced files after doing this
    ds = gdal.BuildVRT(out_file, vrts, separate=True, allowProjectionDifference=False)
    ds.FlushCache()
    return ds


def generate_daily_fire_footprint(inc_id, ref_ds, fire_obs, fbbox, ref_scale, path):

    # get all fire observations associated with the given incident
    this_fire = fire_obs[fire_obs['INCIDENT_IDENTIFIER'] == inc_id]

    fire_pixel_rescale = ref_scale / 0.2
    size = (ref_ds.RasterYSize, ref_ds.RasterXSize)
    size = np.round(fire_pixel_rescale * np.asarray(size)).astype(int)

    days = list(this_fire['ACQ_DATE'].drop_duplicates())
    if len(days) == 0:
        return 0
    first_day = dt.datetime.strptime(days[0], '%Y-%m-%d')
    last_day = dt.datetime.strptime(days[-1], '%Y-%m-%d')
    delta = last_day - first_day
    all_days = [first_day + dt.timedelta(days=i) for i in range(-1, delta.days + 2)]
    all_days = [day.strftime('%Y-%m-%d') for day in all_days]

    drv = gdal.GetDriverByName("GTiff")
    width = int(size[1])
    height = int(size[0])
    ds = drv.Create(f'{path}/{inc_id}/unscaled_fire_footprint.tif', width, height, len(all_days), gdal.GDT_Float32)
    # re-format the bounding box
    fbbox2 = [[fbbox[1], fbbox[3]], [fbbox[0], fbbox[2]]]

    for i, day in enumerate(all_days):
        if i == 0:
            ex_fire_day = this_fire.query('ACQ_DATE == @day & ACQ_TIME < 1200')
        else:
            last_day = all_days[i - 1]
            ex_fire_day = this_fire.query('(ACQ_DATE == @day & ACQ_TIME < 1200) |'
                                          ' (ACQ_DATE == @last_day & ACQ_TIME >= 1200) ')
        z, y, x = np.histogram2d(ex_fire_day['LATITUDE'], ex_fire_day['LONGITUDE'],
                                 bins=size, range=fbbox2)
        z = np.flip(z, axis=0)

        raster = ds.GetRasterBand(i + 1)
        raster.WriteArray(z)
        raster.SetDescription(day)
        tr = np.asarray(ref_ds.GetGeoTransform())
        tr[1] /= fire_pixel_rescale
        tr[5] /= fire_pixel_rescale

        ds.SetGeoTransform(tuple(tr))

        tr = ref_ds.GetGeoTransform()
    ds2 = gdal.Warp(f'{path}/{inc_id}/daily_fire_footprint.vrt', ds, format='VRT', xRes=abs(tr[1]), yRes=abs(tr[5]),
                    resampleAlg='cubic', outputBounds=fbbox)
    del ds2

    return 1


def get_grid_bounds_from_geo_ds(ds):
    tr = ds.GetGeoTransform()
    bounds = [tr[0], tr[3] + ds.RasterYSize * tr[5], tr[0] + ds.RasterXSize * tr[1], tr[3]]
    return tr[1], -tr[5], bounds


def process_single_fire_data(in_dir, out_dir, hrrr_path, inc_id, time_fxx_list):
    # only do this if we have a fire footprint
    if not os.path.isfile(f'{in_dir}/{inc_id}/daily_fire_footprint.vrt'):
        return

    # make directories if needed
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(f'{out_dir}/{inc_id}'):
        os.mkdir(f'{out_dir}/{inc_id}')

    static_data = gdal.Open(f'{in_dir}/{inc_id}/static_data.vrt')

    # make it an array and check we read it right
    bands = [static_data.GetRasterBand(i + 1) for i in range(static_data.RasterCount)]
    np_static_data = np.dstack([band.ReadAsArray() for band in bands])
    if np.all(np_static_data[:, :, -1] == -9999):
        raise ValueError(f'Static data read in wrong: {inc_id}')

    # save to numpy
    np.save(f'{out_dir}/{inc_id}/static_data.npy', np_static_data)

    # save json with documentation of the layers
    static_data_lookup = pd.read_csv('data/input/static_data_params.csv')
    static_data_lookup = dict(zip(static_data_lookup['layer'], static_data_lookup.index))
    with open(f'{out_dir}/{inc_id}/static_data.json', 'w') as f:
        json.dump(static_data_lookup, f)

    # read geotransform info from static data, to be used to read weather data later
    lon_res, lat_res, fire_bbox = get_grid_bounds_from_geo_ds(static_data)

    # read fire footprint data, get number of days
    fire_footprint = gdal.Open(f'{in_dir}/{inc_id}/daily_fire_footprint.vrt')
    num_days = fire_footprint.RasterCount

    # for each fire day
    for fire_day in range(1, num_days + 1):

        # make numbered subdirectory
        if not os.path.isdir(f'{out_dir}/{inc_id}/{fire_day}'):
            os.mkdir(f'{out_dir}/{inc_id}/{fire_day}')

        # get the raster band and the day
        fire = fire_footprint.GetRasterBand(fire_day)
        day = fire.GetDescription()
        day = day.replace('-', '')

        # for the first day only, save some global info about the fire
        if fire_day == 1:
            with open(f'{out_dir}/{inc_id}/info.json', 'w') as f:
                json.dump({'transform': static_data.GetGeoTransform(),
                           'shape': np_static_data.shape[0:2],
                           'start_day': day,
                           'num_days': num_days},
                          f)

        # read the fire footprint for the day as an array, and save
        fire = fire.ReadAsArray()
        np.save(f'{out_dir}/{inc_id}/{fire_day}/fire.npy', fire)

        # for each set of time/fxx params in the list
        for d in time_fxx_list:

            # prepare the regex to grab corresponding GRIB file
            time = d['time']
            fxx = d['fxx']
            regex = f'{hrrr_path}/{day}/*{time}*{fxx}*grib2*'

            # read the grib file
            grib = glob.glob(regex)
            grib = [i for i in grib if '.xml' not in i]
            if len(grib) != 1:
                print(regex)
                raise ValueError("Not exactly one GRIB file found")
            grib_file = grib[0]
            ds = gdal.Open(grib_file)

            # slice and warp to fire bounds using info from static data
            warped = gdal.Warp('temp.vrt', ds, format='VRT', dstSRS='WGS84', xRes=lon_res, yRes=lat_res,
                               resampleAlg='bilinear', outputBounds=fire_bbox)

            # read weather data to array and save
            np_weather_data = np.dstack([warped.GetRasterBand(i + 1).ReadAsArray() for i in range(warped.RasterCount)])
            np.save(f'{out_dir}/{inc_id}/{fire_day}/weather_{time}_{fxx}.npy', np_weather_data)

            # read and save info about the channels
            with open(f'{hrrr_path}/{day}/band_lookup_{time}.json', 'r') as f:
                band_lookup = json.load(f)
            weather_lookup = {key: band_lookup[key] - 1 for key in band_lookup}
            with open(f'{out_dir}/{inc_id}/{fire_day}/weather_{time}.json', 'w') as f:
                json.dump(weather_lookup, f)

    os.remove('temp.vrt')


def make_gifs(path):

    fig = plt.figure()

    for inc_path in glob.glob(f'{path}/*'):

        ds_path = f'{inc_path}/daily_fire_footprint.vrt'
        if os.path.isfile(ds_path):

            ds = gdal.Open(ds_path)
            arr = np.asarray([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(ds.RasterCount)])

            for i, day in enumerate(arr):
                plt.imshow(day)
                plt.set_cmap("hot")
                plt.clim(0, 1)
                plt.colorbar()
                title = ds.GetRasterBand(i + 1).GetDescription()
                plt.title(title)
                plt.savefig(f'{inc_path}/{str(i).zfill(4)}.png')
                plt.clf()

            fp_in = f"{inc_path}/*.png"
            fp_out = f"{inc_path}/daily_fire_footprint.gif"

            pngs = sorted(glob.glob(fp_in))
            img, *imgs = [Image.open(f) for f in pngs]

            duration = [200 for f in pngs]
            duration[-1] = 1000
            img.save(fp=fp_out, format='GIF', append_images=imgs,
                     save_all=True, duration=duration, loop=0)

            rm_files = [os.remove(f) for f in pngs]
    return 1

