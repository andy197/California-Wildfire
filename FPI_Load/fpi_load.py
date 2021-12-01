import rasterio
from pyproj import Transformer
import os
import urllib.request
import logging
import zipfile
import boto3
import psycopg2
import json
from multiprocessing import Pool
import itertools
import shutil
import time

debug = False
secret = json.loads(boto3.client("secretsmanager", region_name="us-east-2").get_secret_value(
        SecretId='CSE6242')["SecretString"])


class LatLonParser:
    # Built by referencing
    # "https://geohackweek.github.io/raster/04-workingwithrasters/"
    # and "https://rasterio.readthedocs.io/en/latest/quickstart.html"

    '''
    Calculates our lat lon data from the image for the fpi value
    '''
    def __init__(self, dataset, target_crs="epsg:4326"):
        self.dataset = dataset
        self.transformer = Transformer.from_crs(dataset.crs, target_crs)

    def convert(self, row, col):
        east, north = self.dataset.xy(row, col)  # image --> spatial coordinates
        lon, lat = self.transformer.transform(east, north)

        return lon, lat


known_values = {
    251: ["dark grey", [4366, 28], (47.29151379967935, -68.26518144482685), [251]], # Ag?
    254: ["light blue", [4370, 21], (47.334197838721295, -68.17850286703049), [251]], # Water?
    252: ["blue gray", [474, 112], (48.79546099020077, -121.85883502773216), [8]], # Wetlands?
    233: ["red", [494, 112], (48.84416738047448, -121.59405089386517), [8]], # Unsure - singular point in sea of green - unlikely an FPI value
    243: ["red", [257, 121], (48.15435816287835, -124.66417159512324), [8]], # Another red dot in a sea of grean / blue - unlikely and FPI value
    253: ["blue green", [310, 151], (48.04354532436763, -123.85241860772184), [8]], # Water?
    249: ["black", [4501, 190], (45.50330102423008, -67.4551529621394), [251]], # In water
    250: ["white", [527, 192], (48.23493480964628, -120.86317124083006), [8]], # Cloud
    235: ["red", [943, 263], (48.45613812825031, -115.12835759304453), [8]], # red dot in green
    192: ["red", [952, 313], (48.031435030065424, -114.87874734476802), [8]], # another red in the middle of green
    190: ["red", [955, 315], (48.01883049559543, -114.83386900714234), [8]], # another red dot, starting to think these are weather stations
    245: ["red", [1018, 330], (47.98835432622254, -113.96119606925049), [8]], # red again
    240: ["red", [4372, 350], (44.64380076573953, -69.78361225499873), [251]],
}


def process_image(path, year, conn):
    print("Processing {}".format(path))
    
    # TODO - Remove breaks
    dates = os.path.split(path)[-1].split(".")[0].split("_")
    first_date = "{}-{}-{}".format(dates[-2][:4], dates[-2][4:6], dates[-2][6:])
    second_date = "{}-{}-{}".format(dates[-1][:4], dates[-1][4:6], dates[-1][6:])
    with rasterio.open(path) as dataset:
        # Create the thing to convert the coordinates
        transformer = LatLonParser(dataset)

        # Read the actual data
        data = dataset.read(1)
        
        # FLAGS
        keep_every = 5
        divisor = 10
        count = 0
        total = data.shape[0] * data.shape[1]
        fraction = int(total/divisor)
        print(total)

        # Keep track of all of the unique pixel values for validation
        rows = []
        lat_col = None
        now = time.time()
        for i, row in enumerate(data):
            for j, col in enumerate(row):
                count+=1
                if count%fraction == 0:
                    load_data(rows, year, conn)
                    rows = []
                    
                    time2 = time.time()
                    print("{}% completed in {} seconds".format(round(100*count/total), round(time2-now)))
                    now = time2
                
                # Pixel coords
                x = j
                y = i
                
                if (lat_col is not None and j > lat_col) or i%keep_every != 0 or j%keep_every != 0:
                    continue

                # Lat/Lon Calc
                # NOTE: THIS IS WRONG - LON AND LAT SHOULD BE FLIPPED - HAS BEEN ACCOUNTED FOR IN THE DB
                lon, lat = transformer.convert(i, j)
                if lat > -115:
                    lot_col = j
                    continue

                # Measured Value
                FPI = int(col)

                # Other Calc
                for val in dataset.sample([(i, j)]):
                    other_val = int(val[0])

                data_row = [x, y, lon, lat, FPI, first_date, second_date, other_val]

                rows.append(data_row)
                
                
                if debug:
                    break
            if debug:
                break
        if len(rows) != 0:
            load_data(rows, year, conn)
            rows = []

        return rows


def download_from_url(url, path):
    print("Downloading {} to {}".format(url, path))
    urllib.request.urlretrieve(url, path)
    
    
def process_zip_file(input_val):
    zipfolder = input_val[0]
    downloaded_full_unzipped_path = input_val[1]
    year = input_val[2]
    
    sub_downloaded_full_unzipped_path_zipped = os.path.join(downloaded_full_unzipped_path, zipfolder)
    sub_downloaded_full_unzipped_path_unzipped = sub_downloaded_full_unzipped_path_zipped.replace(".zip", "")
    
    # MULTIPROCESS HERE
    conn = get_conn()
    if ".zip" in sub_downloaded_full_unzipped_path_zipped:
        with zipfile.ZipFile(sub_downloaded_full_unzipped_path_zipped, 'r') as zip_ref:
            zip_ref.extractall(sub_downloaded_full_unzipped_path_unzipped)
    for datafile in os.listdir(sub_downloaded_full_unzipped_path_unzipped):
        full_data_path = os.path.join(sub_downloaded_full_unzipped_path_unzipped, datafile)
        if full_data_path[-5:] == ".tiff":
            print("Extracting data")
            rows = process_image(full_data_path, year, conn)
    conn.close()
    return "success"


def process_zip(downloaded_full_zip_path, year):
    print("Deleting previous year: {}".format(year))
    delete_year(year)
    
    print("Processing files in {}".format(downloaded_full_zip_path))
    downloaded_full_unzipped_path = downloaded_full_zip_path.replace(".zip", "")
    with zipfile.ZipFile(downloaded_full_zip_path, 'r') as zip_ref:
        zip_ref.extractall(downloaded_full_unzipped_path)
        
    # TODO - Change file limit
    zipfolders = os.listdir(downloaded_full_unzipped_path)
    if debug:
        zipfolders = zipfolders[:5]
        print(zipfolders)
    
    print("Starting multiprocessing - will be processing {} files for the year {}".format(len(zipfolders), year))
    
    if not debug:
        pool = Pool(16)
        r = pool.map_async(process_zip_file, zip(zipfolders, itertools.repeat(downloaded_full_unzipped_path), itertools.repeat(year)))
        r.wait()
    else:
        for input_val in zip(zipfolders, itertools.repeat(downloaded_full_unzipped_path), itertools.repeat(year)):
            process_zip_file(input_val)
        

    shutil.rmtree(downloaded_full_unzipped_path)


def main():
    fpi_template = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/firedanger/download-tool/bulk_download/yearly_bundle/{year}/{year}_Fire_Potential_Index_DATA.zip'
    large_fire_template = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/firedanger/download-tool/bulk_download/yearly_bundle/{year}/{year}_Large_Fire_Probability_DATA.zip'
    expected_fire_template = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/firedanger/download-tool/bulk_download/yearly_bundle/{year}/{year}_Expected_Number_of_Fires_per_Predictive_Service_Area_DATA.zip'

    for year in reversed(range(2001, 2016)):
        print("Processing {}".format(year))
        path = "/tmp/fpi.zip"
        fpi_url = fpi_template.format(year=year)
        
        if not debug:
            download_from_url(fpi_url, path)

        process_zip(path, year)
        

def delete_year(year):
    conn = get_conn()
    curr = conn.cursor()
    
    # Delete matching records
    template = '''
DELETE FROM staging.fpi where year = {}
    '''.format(year)
    # print("DELETING: {}".format(curr.mogrify(template, (rows[0][-3], rows[0][-2]))))
    curr.execute(template)
    conn.commit()


def get_conn():
    conn = psycopg2.connect(
        user=secret["username"],
        password=secret["password"],
        host=secret["host"],
        port=secret["port"],
        database=secret["dbname"]
    )
    
    return conn
    

def load_data(rows, year, conn):
    curr = conn.cursor()
    
    # Load the new records
    template = '''
    INSERT INTO staging.fpi (pixel_x, pixel_y, lng, lat, FPI, first_date, second_date, other, year)
    VALUES {}
    '''.format(','.join(curr.mogrify("(%s, %s, %s, %s, %s, %s, %s, %s, {})".format(year), val).decode("utf-8") for val in rows))
    # print("INSERTING: {}".format(template))
    
    curr.execute(template)
    conn.commit()
    
    curr.close()
    

if __name__ == "__main__":
    main()

