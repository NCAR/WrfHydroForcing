# Quick and dirty program to pull down operational 
# conus MRMS Radar Quality Index data. 

# Logan Karsten
# National Center for Atmospheric Research
# Research Applications Laboratory

import datetime
import urllib
from urllib import request
import http
from http import cookiejar
import os
import sys

outDir = "/glade/p/cisl/nwc/karsten/NWM_v21_Dev/INPUT/MRMS/RadarQualityIndex"
dNow = datetime.datetime.utcnow()
lookBackHours = 24
dTmp = datetime.datetime(dNow.year,dNow.month,dNow.day,dNow.hour)
begDate = dTmp - datetime.timedelta(seconds=3600*lookBackHours)
endDate = dNow

dtProc = endDate - begDate
nStepsProc = dtProc.days*24 + dtProc.seconds/3600.0
ncepHTTP = "https://mrms.ncep.noaa.gov/data/2D/RadarQualityIndex"

for stepTmp in range(0,int(nStepsProc)):
	dCycle = begDate + datetime.timedelta(seconds=3600*stepTmp)
	print("Current Step = " + dCycle.strftime('%Y-%m-%d %H'))

	fileDownload = "MRMS_RadarQualityIndex_00.00_" + dCycle.strftime('%Y%m%d') + \
                       "-" + dCycle.strftime('%H') + '0000.grib2.gz'
	url = ncepHTTP + "/" + fileDownload
	outFile = outDir + "/" + fileDownload
	if os.path.isfile(outFile):
		continue
	try:
		request.urlretrieve(url,outFile)
	except:
		print("FILE: " + url + " not found.")
		continue

