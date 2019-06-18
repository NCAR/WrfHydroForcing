# Quick and dirty program to pull down operational 
# conus Rapid Refresh data (surface files).

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

outDir = "/glade/p/cisl/nwc/karsten/NWM_v21_Dev/INPUT/RAP_Conus"
lookBackHours = 30
cleanBackHours = 240
cleanBackHours2 = 72
lagBackHours = 1 # Wait at least this long back before searching for files. 
dNowUTC = datetime.datetime.utcnow()
dNow = datetime.datetime(dNowUTC.year,dNowUTC.month,dNowUTC.day,dNowUTC.hour)

ncepHTTP = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod"

for hour in range(cleanBackHours,cleanBackHours2,-1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

	# Compose path to directory containing data.
	rapCleanDir = outDir + "/rap." + dCurrent.strftime('%Y%m%d')

	# Check to see if directory exists. If it does, remove it. 
	if os.path.isdir(rapCleanDir):
		print("Removing old RAP data from: " + rapCleanDir)
		shutil.rmtree(rapCleanDir)

# Now that cleaning is done, download files within the download window. 
for hour in range(lookBackHours,lagBackHours,-1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

	rapOutDir = outDir + "/rap." + dCurrent.strftime('%Y%m%d')
	if not os.path.isdir(rapOutDir):
		os.mkdir(rapOutDir)

	if dCurrent.hour == 3 or dCurrent.hour == 9 or dCurrent.hour == 15 or dCurrent.hour == 21:
		# RAP cycles every six hours produce forecasts out to 39 hours.
		nFcstHrs = 39
	else:
		# Otherwise, 21 hour forecasts. 
		nFcstHrs = 21

	for hrDownload in range(0,nFcstHrs+1):
		httpDownloadDir = ncepHTTP + "/rap." + dCurrent.strftime('%Y%m%d')
		fileDownload = "rap.t" + dCurrent.strftime('%H') + \
					   "z.awp130bgrbf" + str(hrDownload).zfill(2) + ".grib2"
		url = httpDownloadDir + "/" + fileDownload
		outFile = rapOutDir + "/" + fileDownload
		if not os.path.isfile(outFile):
			try:
				print('Pulling RAP file: ' + fileDownload + ' from NOMADS')
				request.urlretrieve(url,outFile)
			except:
				print("Unable to retrieve: " + url)
				print("Data may not be available yet...")
				continue

