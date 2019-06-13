# Quick and dirty program to pull down operational 
# CFSv2 forecast data for each ensemble member, for
# each six hour forecast going out to 30 days. 

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

# Program parameters
msgFlag = 1 # 1 = Print to screen, 0 = Do not print unecessary information
outDir = "/glade/p/cisl/nwc/karsten/NWM_v21_Dev/INPUT/CFSv2"
tmpDir = "/glade/scratch/karsten"
lookBackHours = 240 # How many hours to look for data.....
cleanBackHours = 720 # Period between this time and the beginning of the lookback period to cleanout old data
lagBackHours = 120 # Wait at least this long back before searching for files. 

# Define communication of issues. 
emailAddy = 'karsten@ucar.edu'
errTitle = 'Error_get_CFSv2'
warningTitle = 'Warning_get_CFSv2'

pid = os.getpid()
lockFile = tmpDir + "/GET_CFSV2.lock"

# First check to see if lock file exists, if it does, throw error message as
# another pull program is running. If lock file not found, create one with PID.
if os.path.isfile(lockFile):
	fileLock = open(lockFile,'r')
	pid = fileLock.readline()
	warningMsg =  "WARNING: Another CFSv2 Fetch Program Running. PID: " + pid
	warningOut(warningMsg,warningTitle,emailAddy,lockFile)
else:
	fileLock = open(lockFile,'w')
	fileLock.write(str(os.getpid()))
	fileLock.close()

dNowUTC = datetime.datetime.utcnow()
dNow = datetime.datetime(dNowUTC.year,dNowUTC.month,dNowUTC.day,dNowUTC.hour)
fcstHrsDownload = 726
ensNum = "01"
ncepHTTP = "https://nomads.ncdc.noaa.gov/modeldata/cfsv2_forecast_6-hourly_9mon_flxf"

for hour in range(cleanBackHours,lookBackHours,-1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

	# Go back in time and clean out any old data to conserve disk space.
	if dCurrent.hour != 0 and dCurrent.hour != 6 and dCurrent.hour != 12 and dCurrent.hour != 18:
		continue # This is not a CFS cycle hour.
	else:
		# Compose path to directory containing data.
		cfsCleanDir = outDir + "/cfs." + dCurrent.strftime('%Y%m%d') + "/" + \
					  dCurrent.strftime('%H') + "/6hrly_grib_" + ensNum

		# Check to see if directory exists. If it does, remove it.
		if os.path.isdir(cfsCleanDir):
			#print("Removing old CFS data from: " + cfsCleanDir)
			shutil.rmtree(cfsCleanDir)

		# If the subdirectory is empty, remove it.
		cfsCleanDir = outDir + "/cfs." + dCurrent.strftime('%Y%m%d') + "/" + \
					  dCurrent.strftime('%H')

		if os.path.isdir(cfsCleanDir):
			if len(os.listdir(cfsCleanDir)) == 0:
				#print("Removing empty directory: " + cfsCleanDir)
				shutil.rmtree(cfsCleanDir)

		cfsCleanDir = outDir + "/cfs." + dCurrent.strftime('%Y%m%d')

		if os.path.isdir(cfsCleanDir):
			if len(os.listdir(cfsCleanDir)) == 0:
				#print("Removing empty directory: " + cfsCleanDir)
				shutil.rmtree(cfsCleanDir)

# Now that cleaning is done, download files within the download window. 
for hour in range(lookBackHours,lagBackHours,-1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

	if dCurrent.hour != 0 and dCurrent.hour != 6 and dCurrent.hour != 12 and dCurrent.hour != 18:
		continue # THis is not a GFS cycle hour.
	else:
		cfsOutDir1 = outDir + "/cfs." + dCurrent.strftime('%Y%m%d')
		if not os.path.isdir(cfsOutDir1):
			os.mkdir(cfsOutDir1)

		cfsOutDir2 = outDir + "/cfs." + dCurrent.strftime('%Y%m%d') + "/" + \
					 dCurrent.strftime('%H')
		if not os.path.isdir(cfsOutDir2):
			os.mkdir(cfsOutDir2)

		cfsOutDir = outDir + "/cfs." + dCurrent.strftime('%Y%m%d') + "/" + \
					dCurrent.strftime('%H') + "/6hrly_grib_" + ensNum

		httpDownloadDir = ncepHTTP + "/" + dCurrent.strftime('%Y') + "/" + \
						  dCurrent.strftime('%Y%m') + "/" + \
						  dCurrent.strftime('%Y%m%d') + "/" + \
						  dCurrent.strftime('%Y%m%d%H')
		if not os.path.isdir(cfsOutDir):
			os.mkdir(cfsOutDir)
		# Download hourly files from NCEP to hour 120.
		for hrDownload in range(0,fcstHrsDownload,6):
			dCurrent2 = dCurrent + datetime.timedelta(seconds=3600*hrDownload)
			fileDownload = "flxf" + dCurrent2.strftime('%Y%m%d%H') + \
						   "." + ensNum + "." + dCurrent.strftime('%Y%m%d%H') + ".grb2"
			url = httpDownloadDir + "/" + fileDownload
			outFile = cfsOutDir + "/" + fileDownload
			if not os.path.isfile(outFile):
				try:
					#print('Pulling CFS file: ' + fileDownload + ' from NOMADS')
					request.urlretrieve(url,outFile)
				except:
					#print("Unable to retrieve: " + url)
					#print("Data may not be available yet...")
					continue
