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

# Program parameters
msgFlag = 1 # 1 = Print to screen, 0 = Do not print unecessary information
outDir = "/glade/p/cisl/nwc/karsten/NWM_v21_Dev/INPUT/MRMS/RadarQualityIndex"
tmpDir = "/glade/scratch/karsten"
lookBackHours = 72 # How many hours to look for data.....
cleanBackHours = 240 # Period between this time and the beginning of the lookback period to cleanout old data
lagBackHours = 1 # Wait at least this long back before searching for files.
dNowUTC = datetime.datetime.utcnow()
dNow = datetime.datetime(dNowUTC.year,dNowUTC.month,dNowUTC.day,dNowUTC.hour)
lookBackHours = 24
ncepHTTP = "https://mrms.ncep.noaa.gov/data/2D/RadarQualityIndex"

# Define communication of issues.
emailAddy = 'jon.doe@youremail.com'
errTitle = 'Error_get_MRMS_RQI'
warningTitle = 'Warning_get_MRMS_RQI'

pid = os.getpid()
lockFile = tmpDir + "/GET_MRMS_RQI.lock"

# First check to see if lock file exists, if it does, throw error message as
# another pull program is running. If lock file not found, create one with PID.
if os.path.isfile(lockFile):
	fileLock = open(lockFile,'r')
	pid = fileLock.readline()
	warningMsg =  "WARNING: Another MRMS RQI Fetch Program Running. PID: " + pid
	warningOut(warningMsg,warningTitle,emailAddy,lockFile)
else:
	fileLock = open(lockFile,'w')
	fileLock.write(str(os.getpid()))
	fileLock.close()

for hour in range(cleanBackHours, lookBackHours, -1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600 * hour)

	# Compose path to MRMS file to clean.
	fileClean = outDir + "/MRMS_RadarQualityIndex_00.00_" + dCurrent.strftime('%Y%m%d') + \
				"-" + dCurrent.strftime('%H') + '0000.grib2.gz'

	if os.path.isfile(fileClean):
		print("Removing old file: " + fileClean)
		os.remove(fileClean)

for hour in range(lookBackHours,lagBackHours,-1):
	dCycle = dNow - datetime.timedelta(seconds=3600*hour)
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

# Remove the LOCK file.
os.remove(lockFile)

