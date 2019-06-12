# Quick and dirty program to pull down operational 
# 3-km NAM Nest Hawaii data in GRIB2 format.

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
import math

outDir = "/glade/p/cisl/nwc/karsten/NWM_v21_Dev/INPUT/NAM_Nest_Alaska"
lookBackDays = 5
dCurrent = datetime.datetime.utcnow()
lastCycleHour = int((math.floor(dCurrent.hour/6))*6.0)
lastCycleDate = datetime.datetime(dCurrent.year,dCurrent.month,dCurrent.day,lastCycleHour)
firstCycleDate = lastCycleDate - datetime.timedelta(seconds=3600*24*lookBackDays)

dtProc = lastCycleDate - firstCycleDate
nCycles = (dtProc.days*24 + dtProc.seconds/3600.0)/6.0
ncepHTTP = "https://ftp.ncep.noaa.gov/data/nccf/com/nam/prod"

for cycle in range(0,int(nCycles)):
	dCycle = firstCycleDate + datetime.timedelta(seconds=3600*cycle*6.0)
	print("Current Cycle = " + dCycle.strftime('%Y-%m-%d %H'))

	namOutDir = outDir + "/nam." + dCycle.strftime('%Y%m%d')	

	httpDownloadDir = ncepHTTP + "/nam." + dCycle.strftime('%Y%m%d')
	if not os.path.isdir(namOutDir):
		os.mkdir(namOutDir)
	# Download hourly files from NCEP to hour 120.
	for hrDownload in range(0,61):
		fileDownload = "nam.t" + dCycle.strftime('%H') + \
		               "z.alaskanest.hiresf" + str(hrDownload).zfill(2) + \
			       ".tm00.grib2"
		url = httpDownloadDir + "/" + fileDownload
		print(url)
		outFile = namOutDir + "/" + fileDownload
		request.urlretrieve(url,outFile)
