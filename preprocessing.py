import pyodbc
import geopandas as gpd
import pandas as pd
import numpy as np
import datetime as dt
from dateutil import parser
from shapely import geometry
from herbie.archive import Herbie
import matplotlib.pyplot as plt
import pickle
import geopy.distance as dst


def slice_df(df, field, lower_bound, upper_bound):
    """Slices a DataFrame on prescribed field, keeping all elements strictly between two bounds."""

    return df[(df[field] > lower_bound) & (df[field] < upper_bound)]


def from_reference(time, reference_time=dt.datetime(2000, 1, 1, 0, 0)):
    """Transforms a datetime object into a float, the number of seconds after a given reference time.
    Used to simplify logical operations with times."""

    return (time - reference_time).total_seconds()


def ref_to_datetime(time_from_reference, reference_time=dt.datetime(2000, 1, 1, 0, 0)):
    """Turns float into datetime object by adding seconds to a given reference time. Inverse of from_reference."""

    return reference_time + dt.timedelta(seconds=time_from_reference)


def clean_fire_pixels(in_path):
    """Preliminary cleaning of VIIRS fire pixels.

    Arguments:
        in_path: file name of raw shapefile with VIIRS data points

    Output:
        pandas DataFrame with fire pixels, including a unique pixel_id for each observation.
    """

    # read in raw VIIRS GeoDataFrame, drop shapely Points
    fires_gdf = gpd.read_file(in_path)
    fires_df = fires_gdf.drop(columns='geometry')

    # get time in float format
    fires_df['time'] = fires_df.apply(lambda x: parser.parse(x['ACQ_DATE'] + ' ' + x['ACQ_TIME']), axis=1)
    fires_df['time_from_reference'] = fires_df['time'].apply(from_reference)

    # create unique pixel ids based on time captured, numbered in increasing order of brightness
    fires_df['pixel_id'] = fires_df.apply(lambda x: x['ACQ_DATE'].replace('-', '') + '_' + x['ACQ_TIME'], axis=1)
    fires_df = fires_df.sort_values(by=['pixel_id', 'BRIGHTNESS'])
    fires_df['pixel_id'] = fires_df['pixel_id'] + '_' + fires_df.groupby('pixel_id').cumcount().astype(str).str.zfill(6)

    # re-order columns to put id first
    cols = ['pixel_id', 'LATITUDE', 'LONGITUDE', 'BRIGHTNESS', 'SCAN', 'TRACK', 'ACQ_DATE',
            'ACQ_TIME', 'SATELLITE', 'INSTRUMENT', 'CONFIDENCE', 'VERSION',
            'BRIGHT_T31', 'FRP', 'DAYNIGHT', 'TYPE', 'time', 'time_from_reference']
    fires_df = fires_df[cols]

    # return
    return fires_df


def merge_final_boundaries_to_sit_incidents(sit_conn_str, year, boundaries_path):

    # connect to Microsoft Access Database and read the incidents and reports tables
    conn = pyodbc.connect(sit_conn_str)
    incidents_df = pd.read_sql('select * from SIT209_HISTORY_INCIDENTS', conn)
    reports_df = pd.read_sql('select * from SIT209_HISTORY_INCIDENT_209_REPORTS', conn)

    # filter the incidents DataFrame to only include incidents where there are SIT reports
    incidents_w_reports = reports_df['INC_IDENTIFIER'].unique()
    incidents_df['INCIDENT_NAME'] = incidents_df['INCIDENT_NAME'].str.upper()
    incidents = incidents_df[incidents_df['INCIDENT_IDENTIFIER'].isin(incidents_w_reports)].copy()

    # turn the point of origin into a Shapely point, currently unused but keeping it here in case it is needed
    # to merge in a smarter way (using geography)
    incidents['POO_LATITUDE'] = incidents['POO_LATITUDE'].astype(float)
    incidents['POO_LONGITUDE'] = incidents['POO_LONGITUDE'].astype(float)
    incidents['POO'] = incidents.apply(lambda x: geometry.Point(x['POO_LONGITUDE'], x['POO_LATITUDE']), axis=1)

    # clean some columns, including IRWINID, which is used to merge with data set of final fire boundaries
    cols = ['INCIDENT_IDENTIFIER', 'INCIDENT_NAME', 'DISCOVERY_DATE', 'POO', 'IRWIN_IDENTIFIER']
    incidents = incidents[cols]
    incidents = incidents.rename(columns={'IRWIN_IDENTIFIER': 'IRWINID'})

    # read in GeoDataFrame of final fire boundaries, restrict to given year
    boundaries = gpd.read_file(boundaries_path)
    boundaries = boundaries[boundaries['FIRE_YEAR'] == str(year)].copy()

    # filter out very small fires (<1 acre if I recall correctly), not sure why I did this or if it helps
    boundaries['area'] = boundaries['geometry'].apply(lambda x: x.area)
    boundaries = boundaries[boundaries['area'] >= 1e-7]

    # only keep fires with IRWINID, clean up format
    boundaries = boundaries[boundaries['IRWINID'].notna()]
    boundaries['IRWINID'] = boundaries['IRWINID'].astype(str).str.upper()
    boundaries['IRWINID'] = boundaries['IRWINID'].apply(
        lambda s: ''.join(x for x in s if x.isalpha() or x.isdigit() or x == '-'))
    cols = ['FID', 'MAP_METHOD', 'IRWINID', 'INCIDENT', 'UNIT_ID', 'POO_RESP_I', 'geometry', 'area']
    boundaries = boundaries[cols]

    # if there are multiple boundaries associated with the same IRWINID, keep the largest area
    boundaries = boundaries.sort_values('area', ascending=False).drop_duplicates('IRWINID', keep='first')

    # merge in SIT incident id where we have matching IRWINID's, drop all boundaries without merge
    boundaries = pd.merge(incidents, boundaries, on='IRWINID', how='right')
    boundaries = boundaries[boundaries['INCIDENT_IDENTIFIER'].notna()].copy()

    # get the bounding box of each fire boundary
    boundaries['bbox'] = boundaries['geometry'].apply(lambda x: x.bounds)

    # return
    return boundaries


def merge_fire_pixels_to_boundaries(boundaries_df, fire_pixels_df, day_offset=10, drop_dups=True):

    dfs = []
    for _, fire in boundaries_df.iterrows():

        bbox = fire['bbox']
        start_time = from_reference(parser.parse(fire['DISCOVERY_DATE'])) - 3600 * 24 * 7

        loc_df = slice_df(fire_pixels_df, 'LONGITUDE', bbox[0], bbox[2])
        loc_df = slice_df(loc_df, 'LATITUDE', bbox[1], bbox[3])
        loc_df = slice_df(loc_df, 'time_from_reference', start_time, np.inf)

        loc_df['last_time'] = loc_df['time_from_reference'].shift()
        loc_df['time_offset'] = loc_df['time_from_reference'] - loc_df['last_time']
        loc_df['long_offset'] = loc_df['time_offset'] > 24 * 3600 * day_offset
        long_offsets = np.where(loc_df['long_offset'])[0]
        if len(long_offsets) > 0:
            loc_df = loc_df.iloc[:long_offsets[0]]

        loc_df['INCIDENT_IDENTIFIER'] = fire['INCIDENT_IDENTIFIER']
        cols = ['pixel_id', 'INCIDENT_IDENTIFIER']
        dfs.append(loc_df[cols].copy())

    out_df = pd.concat(dfs)

    if drop_dups:
        dups = dict(out_df['pixel_id'].value_counts() > 1)
        dups = [k for k in dups if dups[k]]
        overlaps = out_df[out_df['pixel_id'].isin(dups)]['INCIDENT_IDENTIFIER'].unique()
        out_df = out_df[~out_df['INCIDENT_IDENTIFIER'].isin(overlaps)]

    return out_df


def get_fire_pixels(viirs_path, sit_conn_str, boundaries_path, year):

    clean_viirs = clean_fire_pixels(viirs_path)
    clean_boundaries = merge_final_boundaries_to_sit_incidents(sit_conn_str, year, boundaries_path)
    merged_pixels = merge_fire_pixels_to_boundaries(clean_boundaries, clean_viirs)

    return clean_viirs, clean_boundaries, merged_pixels


def local_miles_per_lat_lon(bbox, k=100):
    # get lat, lon of center of bbox
    lat = (bbox[3] + bbox[1]) / 2
    lon = (bbox[2] + bbox[0]) / 2

    coords1 = (lat - 1 / k, lon)
    coords2 = (lat + 1 / k, lon)
    miles_per_lat = dst.distance(coords1, coords2).miles * k / 2
    coords1 = (lat, lon - 1 / k)
    coords2 = (lat, lon + 1 / k)
    miles_per_lon = dst.distance(coords1, coords2).miles * k / 2

    return miles_per_lat, miles_per_lon


def get_grid_from_bbox(bbox, tick_miles, buffer_miles=0):
    # locally, find the number of miles per degree latititude and per degree longitude
    miles_per_lat, miles_per_lon = local_miles_per_lat_lon(bbox)

    # sanity check bounds
    lon_length = bbox[2] - bbox[0]
    lat_length = bbox[3] - bbox[1]
    area = lon_length * lat_length * miles_per_lat * miles_per_lon
    if area > 10000:
        raise ValueError('Bounding box more than 10,000 square miles')

    # add in buffer to bbox (buffer added on all sides)
    bbox = list(bbox)
    bbox[0] -= buffer_miles / miles_per_lon
    bbox[2] += buffer_miles / miles_per_lon
    bbox[1] -= buffer_miles / miles_per_lat
    bbox[3] += buffer_miles / miles_per_lat

    num_lon_ticks = (lon_length * miles_per_lon) / tick_miles
    rounded_num_lon_ticks = int(1 + num_lon_ticks)
    lon_buffer = lon_length * (rounded_num_lon_ticks / num_lon_ticks - 1) / 2
    lon_grid = np.linspace(bbox[0] - lon_buffer, bbox[2] + lon_buffer, rounded_num_lon_ticks + 1)

    num_lat_ticks = (lat_length * miles_per_lat) / tick_miles
    rounded_num_lat_ticks = int(1 + num_lat_ticks)
    lat_buffer = lat_length * (rounded_num_lat_ticks / num_lat_ticks - 1) / 2
    lat_grid = np.linspace(bbox[1] - lat_buffer, bbox[3] + lat_buffer, rounded_num_lat_ticks + 1)

    # construct grid
    dfs = []
    for i, lon in enumerate(lon_grid):
        df = pd.DataFrame({'lat': lat_grid, 'y': range(len(lat_grid))})
        df['lon'] = lon
        df['x'] = i
        dfs.append(df)
    grid = pd.concat(dfs)[['x', 'y', 'lon', 'lat']]

    return grid.reset_index(drop=True)


def extract_hrrr(time, fields):
    H = Herbie(time, model='hrrr', product='sfc', fxx=0)

    d = {}
    lat = None
    for _, field in fields.iterrows():

        download_code = field['download_code']
        arr = H.xarray(download_code)
        if lat is None:
            lat = np.ravel(np.asarray(arr['latitude']))
            lon = np.ravel(np.asarray(arr['longitude']))
            d['lat'] = lat
            d['lon'] = lon
        else:
            assert all(lat == np.ravel(np.asarray(arr['latitude'])))
            assert all(lon == np.ravel(np.asarray(arr['longitude'])))
        d[download_code] = np.ravel(np.asarray(arr[field['field_name']]))
    df = pd.DataFrame(d)
    df['lon'] -= 360

    return df


def find_nearest_k_records(target, source, k=4, x='x', y='y'):
    t = target.copy()
    for i, pixel in t.iterrows():

        x_1 = pixel[x]
        y_1 = pixel[y]

        source['sqdist'] = np.power((source[x] - x_1), 2) + np.power((source[y] - y_1), 2)
        s = source.sort_values('sqdist')
        for j in range(k):
            t.loc[i, f'nearest_{j + 1}'] = s.index[j]
            t.loc[i, f'sqdist_{j + 1}'] = s['sqdist'].iloc[j]
    for j in range(k):
        t[f'nearest_{j + 1}'] = t[f'nearest_{j + 1}'].astype(int)

    return t


def get_relevant_lookups(hrrr_example, grid, miles_per_lat, miles_per_lon, tick_miles, buffer_miles=3):
    # slice to relevant region in hrrr data
    lon_min = grid['lon'].min()
    lon_max = grid['lon'].max()
    lat_min = grid['lat'].min()
    lat_max = grid['lat'].max()
    lon_buffer = buffer_miles / miles_per_lon
    lat_buffer = buffer_miles / miles_per_lat
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    bbox_df = slice_df(hrrr_example, 'lon', lon_min, lon_max)
    bbox_df = slice_df(bbox_df, 'lat', lat_min, lat_max)

    bbox_df['x'] = (bbox_df['lon'] - grid['lon'].min()) * miles_per_lon / tick_miles
    bbox_df['y'] = (bbox_df['lat'] - grid['lat'].min()) * miles_per_lat / tick_miles

    knn_lookup = find_nearest_k_records(grid, bbox_df)

    return knn_lookup, bbox_df


def aggregate_knn_data(knn_lookup, source_df, agg_cols,
                       kernel_func=lambda x: 1 / x, weight_col='sqdist'):
    weight_cols = [col for col in knn_lookup.columns if weight_col in col]
    knn_lookup[weight_cols] = knn_lookup[weight_cols].apply(kernel_func)
    knn_lookup['total_weight'] = knn_lookup[weight_cols].sum(axis=1)

    for col in weight_cols:
        knn_lookup[col] = knn_lookup[col] / knn_lookup['total_weight']

    dfs = []
    for i in range(1, 5):
        source_df_merge = source_df[agg_cols].reset_index()
        merged = pd.merge(knn_lookup, source_df_merge, left_on=f'nearest_{i}',
                          right_on='index', how='left')

        for col in agg_cols:
            merged[col] *= merged[f'{weight_col}_{i}']
        dfs.append(merged)

    aggs = {col: sum for col in agg_cols}
    knn = pd.concat(dfs).groupby(['x', 'y'], as_index=False).agg(aggs)

    return knn


def merge_fire_and_weather_data(fire_pixels, boundaries, incident_id, weather_fields, hr_utc=12, tick_miles=0.6,
                                day_buffer=0, time_buffer=3600 * 24, beta=-1):

    # grab the bounding box of the final fire boundary
    bbox = boundaries.set_index('INCIDENT_IDENTIFIER').loc[incident_id, 'bbox']

    # use this to estimate (locally) miles per degree latitude and miles per degree longitude
    miles_per_lat, miles_per_lon = local_miles_per_lat_lon(bbox)

    # make a uniform grid of points in the final boundary bounding box
    grid = get_grid_from_bbox(bbox, tick_miles)

    # restrict fire pixel DataFrame to observations from this fire
    this_fire = fire_pixels[fire_pixels['INCIDENT_IDENTIFIER'] == incident_id]

    # get a list of download codes for HRRR data
    download_codes = weather_fields['download_code'].unique()

    # generate a list of times to extract data, once daily at hr_utc:00 (UTC time)
    first_hrrr = ref_to_datetime(this_fire['time_from_reference'].min())
    first_hrrr = first_hrrr - dt.timedelta(days=1 + day_buffer)
    first_hrrr = first_hrrr.replace(hour=hr_utc, minute=0)
    last_hrrr = ref_to_datetime(this_fire['time_from_reference'].max())
    last_hrrr = last_hrrr + dt.timedelta(days=1 + day_buffer)
    last_hrrr = last_hrrr.replace(hour=hr_utc, minute=0)
    total_days = (last_hrrr - first_hrrr).days
    all_days = [first_hrrr + dt.timedelta(days=i) for i in range(total_days + 1)]

    # initialize some variables
    output = {}
    knn_lookup = None
    ref_bbox = None

    for time in all_days:
        print(dt.datetime.now())
        hrrr = extract_hrrr(time, weather_fields)

        if knn_lookup is None:
            knn_lookup, ref_bbox = get_relevant_lookups(hrrr, grid, miles_per_lat, miles_per_lon, tick_miles)

        bbox_df = hrrr.loc[ref_bbox.index]
        if not all(bbox_df[['lat', 'lon']] == ref_bbox[['lat', 'lon']]):
            raise ValueError('Inconsistent HRRR data indexing')

        merged = aggregate_knn_data(knn_lookup, bbox_df, download_codes)

        slice_time = from_reference(time)
        fire_time_slice = slice_df(this_fire, 'time_from_reference', slice_time - time_buffer, slice_time).copy()

        fire_time_slice['x'] = (fire_time_slice['LONGITUDE'] - grid['lon'].min()) * miles_per_lon / tick_miles
        fire_time_slice['y'] = (fire_time_slice['LATITUDE'] - grid['lat'].min()) * miles_per_lat / tick_miles
        fire_time_slice['x'] = np.round(fire_time_slice['x']).astype(int)
        fire_time_slice['y'] = np.round(fire_time_slice['y']).astype(int)

        test = fire_time_slice.groupby(['x', 'y'], as_index=False).agg({'pixel_id': 'count'})

        test['on_fire'] = 1 - np.exp(beta * test['pixel_id'])
        merged = pd.merge(merged, test[['x', 'y', 'on_fire']], on=['x', 'y'], how='left', validate='1:1')
        assert (merged['on_fire'].notna().sum() == len(test))
        merged['on_fire'] = merged['on_fire'].fillna(0)

        merged = pd.merge(knn_lookup[['x', 'y', 'lat', 'lon']], merged, on=['x', 'y'], how='outer', validate='1:1')
        assert len(merged) == len(knn_lookup)

        output[time] = merged

    return output


