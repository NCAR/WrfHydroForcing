# Quick and dirty program to pull down operational 
# conus HRRR data (surface files).

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
import shutil
import smtplib
from email.mime.text import MIMEText

def errOut(msgContent,emailTitle,emailRec,lockFile):
	msg = MIMEText(msgContent)
	msg['Subject'] = emailTitle
	msg['From'] = emailRec
	msg['To'] = emailRec
	s = smtplib.SMTP('localhost')
	s.sendmail(emailRec,[emailRec],msg.as_string())
	s.quit()
	# Remove lock file
	os.remove(lockFile)
	sys.exit(1)

def warningOut(msgContent,emailTitle,emailRec,lockFile):
	msg = MIMEText(msgContent)
	msg['Subject'] = emailTitle
	msg['From'] = emailRec
	msg['To'] = emailRec
	s = smtplib.SMTP('localhost')
	s.sendmail(emailRec,[emailRec],msg.as_string())
	s.quit()
	sys.exit(1)

def msgUser(msgContent,msgFlag):
	if msgFlag == 1:
		print(msgContent)

outDir = "/glade/p/cisl/nwc/karsten/NWM_v21_Dev/INPUT/HRRR_Conus"
tmpDir = "/glade/scratch/karsten"
lookBackHours = 30
cleanBackHours = 240
cleanBackHours2 = 72
lagBackHours = 1 # Wait at least this long back before searching for files. 
dNowUTC = datetime.datetime.utcnow()
dNow = datetime.datetime(dNowUTC.year,dNowUTC.month,dNowUTC.day,dNowUTC.hour)

#dtProc = endDate - begDate
#nCycles = dtProc.days*24 + dtProc.seconds/3600.0
ncepHTTP = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"

# Define communication of issues.
emailAddy = 'jon.doe@youremail.com'
errTitle = 'Error_get_Conus_HRRR'
warningTitle = 'Warning_get_Conus_HRRR'

pid = os.getpid()
lockFile = tmpDir + "/GET_Conus_HRRR.lock"

for hour in range(cleanBackHours,cleanBackHours2,-1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

	# Compose path to directory containing data.
	hrrrCleanDir = outDir + "/hrrr." + dCurrent.strftime('%Y%m%d')

	# Check to see if directory exists. If it does, remove it. 
	if os.path.isdir(hrrrCleanDir):
		print("Removing old HRRR data from: " + hrrrCleanDir)
		shutil.rmtree(hrrrCleanDir)

# Now that cleaning is done, download files within the download window. 
for hour in range(lookBackHours,lagBackHours,-1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

	hrrrOutDir = outDir + "/hrrr." + dCurrent.strftime('%Y%m%d')
	if not os.path.isdir(hrrrOutDir):
		os.mkdir(hrrrOutDir)

	if dCurrent.hour % 6 == 0:
		# HRRR cycles every six hours produce forecasts out to 36 hours.
		nFcstHrs = 36
	else:
		# Otherwise, 18 hour forecasts. 
		nFcstHrs = 18

	for hrDownload in range(0,nFcstHrs+1):
		httpDownloadDir = ncepHTTP + "/hrrr." + dCurrent.strftime('%Y%m%d') + "/conus"
		fileDownload = "hrrr.t" + dCurrent.strftime('%H') + \
					   "z.wrfsfcf" + str(hrDownload).zfill(2) + ".grib2"
		url = httpDownloadDir + "/" + fileDownload
		outFile = hrrrOutDir + "/" + fileDownload
		if not os.path.isfile(outFile):
			try:
				print('Pulling HRRR file: ' + fileDownload + ' from NOMADS')
				request.urlretrieve(url,outFile)
			except:
				print("Unable to retrieve: " + url)
				print("Data may not be available yet...")
				continue

# Remove the LOCK file.
os.remove(lockFile)
