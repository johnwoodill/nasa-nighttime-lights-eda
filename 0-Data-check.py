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

import gdal, os

FDIR = '/data2/NASA_Nighttime_Lights/'

file = 'h111213v/h11v12/VNP46A2.A2019100.h11v12.001.boat.hdf'

file_loc = FDIR + file

fileExtension = ".tif"

## Open HDF file (first if multiple files are in same directory)
hdflayer = gdal.Open(file_loc, gdal.GA_ReadOnly)

# Open raster layer
for i in range(4):
    rlayer = gdal.Open(hdflayer.GetSubDatasets()[i][0], gdal.GA_ReadOnly)
    outputName = rlayer.GetMetadata_Dict()['long_name']

    outputName = outputName.strip().replace(" ","_").replace("/","_")
    
    outputFolder = "figures/"
    outputRaster = outputFolder + outputName + '.tif'
    gdal.Warp(outputRaster, rlayer)
    
    
