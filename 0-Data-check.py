import pandas as pd
import numpy as np
import glob
import xarray as xr
import os
import multiprocessing
from dask import delayed, compute
from distributed import Client
import gdal
import os


### Convert to tif
# --------------------------------------

import gdal, os, gdalconst

FDIR = '/data2/NASA_Nighttime_Lights/'

file = 'h111213v/h11v12/VNP46A2.A2019100.h11v12.001.boat.hdf'

file_loc = FDIR + file

fileExtension = ".tif"

## Open HDF file (first if multiple files are in same directory)
hdflayer = gdal.Open(file_loc)

hdflayer.GetProjection()

hdflayer.RasterXSize
hdflayer.RasterYSize

ulx, xres, xskew, uly, yskew, yres  = hdflayer.GetGeoTransform()
lrx = ulx + (hdflayer.RasterXSize * xres)
lry = uly + (hdflayer.RasterYSize * yres)

from osgeo import ogr
from osgeo import osr

# Setup the source projection - you can also import from epsg, proj4...
source = osr.SpatialReference()
source.ImportFromWkt(hdflayer.GetProjection())

# The target projection
target = osr.SpatialReference()
target.ImportFromEPSG(4326)

# Create the transform - this can be used repeatedly
transform = osr.CoordinateTransformation(source, target)

# Transform the point. You can also create an ogr geometry and use the more generic `point.Transform()`
transform.TransformPoint(ulx, uly)

# Open raster layer
for i in range(4):
    rlayer = gdal.Open(hdflayer.GetSubDatasets()[i][0], gdal.GA_ReadOnly)
    outputName = rlayer.GetMetadata_Dict()['long_name']

    outputName = outputName.strip().replace(" ","_").replace("/","_")
    
    outputFolder = "figures/"
    outputRaster = outputFolder + outputName + '.tif'
    gdal.Warp(outputRaster, rlayer)
    
    
    
from rastertodataframe import raster_to_dataframe

tif_loc = "figures/Boat_Flag.tif"
df = raster_to_dataframe(tif_loc)
df
df.iloc[:, 0].unique()

import xarray as xr
from affine import Affine
ds = xr.open_rasterio(tif_loc)
dsp = ds.to_dataframe()


da = xr.open_rasterio(tif_loc)
transform = Affine.from_gdal(*da.attrs['transform'])
nx, ny = da.sizes['x'], da.sizes['y']
x, y = np.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5) * transform
len(x)
len(y)

k = 0
for i in range(len(x)):
    for j in range(len(y)):
        lon = x[i]
        lat = y[j]
        boat = df[k]
        k = k + 1
        print(lon, lat, k)
        
        
        
        


from osgeo import gdal
ds = gdal.Open(tif_loc)
width = ds.RasterXSize
height = ds.RasterYSize
gt = ds.GetGeoTransform()
minx = gt[0]
miny = gt[3] + width*gt[4] + height*gt[5] 
maxx = gt[0] + width*gt[1] + height*gt[2]
maxy = gt[3] 





    
### Get values
# ------------------------------
# Process cyano .tif files to get data
def proc_tif(ThisCol, ThisRow, tif_loc):
    tif_loc = "figures/Boat_Flag.tif"
    
    ds = xr.open_dataset(tif_loc)
    
    # Open tif file
    ds       = gdal.OpenShared(tif_loc, gdalconst.GA_ReadOnly)
    GeoTrans = ds.GetGeoTransform()
    
    # Get col/row range
    ColRange = range(ds.RasterXSize)
    RowRange = range(ds.RasterYSize)
    
    # First (only) band of data
    rBand    = ds.GetRasterBand(1) # first band
    
    # specify the centre offset
    HalfX    = GeoTrans[1] / 2
    HalfY    = GeoTrans[5] / 2

    # Get data values (CI)
    ThisCol = 1
    ThisRow = 1
    RowData = rBand.ReadAsArray(0, ThisRow, ds.RasterXSize, 1)[0]

    # Process if CI is greater than zero
    if RowData[ThisCol] >= 0:
        # lon/lat
        X = GeoTrans[0] + ( ThisCol * GeoTrans[1] )
        Y = GeoTrans[3] + ( ThisRow * GeoTrans[5] )
        X += HalfX
        Y += HalfY
        # Return lon, lat, and CI
        return (X, Y, RowData[ThisCol])
    else:
        # Otherwise return None
        return None
    
    
    
    
    
ds       = gdal.OpenShared(tif_loc, gdalconst.GA_ReadOnly)
ColRange = range(ds.RasterXSize)
RowRange = range(ds.RasterYSize)

# Read in data
myarray = ds.GetRasterBand(1).ReadAsArray()

# Only process data that is available
rc = np.where(myarray >= 0)

# Get lon/lat
xx = [x for x in rc[0][:]]
yy = [y for y in rc[1][:]]

# Bind lon/lat
xxyy = pd.DataFrame({'x': xx, 'y': yy})
        

lst = [proc_tif(x, y, tif_loc) for x, y in zip(xxyy['y'], xxyy['x'])]

lon = [x[0] for x in lst]
lat = [x[1] for x in lst]
ci = [x[2] for x in lst]


