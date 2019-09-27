# Quick and dirty program to pull down operational 
# GFS data on the Gaussian grid in GRIB2 format. 

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

outDir = "/glade/p/cisl/nwc/nwm_forcings/Forcing_Inputs/GFS_13km_Global"
tmpDir = "/glade/scratch/karsten"
lookBackHours = 48 # How many hours to look for data.....
cleanBackHours = 240 # Period between this time and the beginning of the lookback period to cleanout old data.  
lagBackHours = 6 # Wait at least this long back before searching for files. 
dNowUTC = datetime.datetime.utcnow()
dNow = datetime.datetime(dNowUTC.year,dNowUTC.month,dNowUTC.day,dNowUTC.hour)
ncepHTTP = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"

# Define communication of issues.
emailAddy = 'jon.doe@youremail.com'
errTitle = 'Error_get_GFS_Full'
warningTitle = 'Warning_get_GFS_Full'

pid = os.getpid()
lockFile = tmpDir + "/GET_GFS_Full.lock"

# First check to see if lock file exists, if it does, throw error message as
# another pull program is running. If lock file not found, create one with PID.
if os.path.isfile(lockFile):
	fileLock = open(lockFile,'r')
	pid = fileLock.readline()
	warningMsg =  "WARNING: Another GFS Global FV3 Fetch Program Running. PID: " + pid
	warningOut(warningMsg,warningTitle,emailAddy,lockFile)
else:
	fileLock = open(lockFile,'w')
	fileLock.write(str(os.getpid()))
	fileLock.close()

for hour in range(cleanBackHours,lookBackHours,-1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

	# Go back in time and clean out any old data to conserve disk space. 
	if dCurrent.hour != 0 and dCurrent.hour != 6 and dCurrent.hour != 12 and dCurrent.hour != 18:
		continue # This is not a GFS cycle hour. 
	else:
		# Compose path to directory containing data. 
		gfsCleanDir = outDir + "/gfs." + dCurrent.strftime('%Y%m%d') + "/" + dCurrent.strftime('%H')

		# Check to see if directory exists. If it does, remove it. 
		if os.path.isdir(gfsCleanDir):
			print("Removing old GFS data from: " + gfsCleanDir)
			shutil.rmtree(gfsCleanDir)

		# Check to see if parent directory is empty.
		gfsCleanDir = outDir + "/gfs." + dCurrent.strftime('%Y%m%d')
		if os.path.isdir(gfsCleanDir):
			if len(os.listdir(gfsCleanDir)) == 0:
				print("Removing empty directory: " + gfsCleanDir)
				shutil.rmtree(gfsCleanDir)

# Now that cleaning is done, download files within the download window. 
for hour in range(lookBackHours,lagBackHours,-1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

	if dCurrent.hour != 0 and dCurrent.hour != 6 and dCurrent.hour != 12 and dCurrent.hour != 18:
		continue # THis is not a GFS cycle hour. 
	else:
		gfsOutDir1 = outDir + "/gfs." + dCurrent.strftime('%Y%m%d')
		if not os.path.isdir(gfsOutDir1):
			print("Making directory: " + gfsOutDir1)
			os.mkdir(gfsOutDir1)

		gfsOutDir2 = gfsOutDir1 + "/" + dCurrent.strftime('%H')
	
		httpDownloadDir = ncepHTTP + "/gfs." + dCurrent.strftime('%Y%m%d') + "/" + dCurrent.strftime('%H')
		if not os.path.isdir(gfsOutDir2):
			print('Making directory: ' + gfsOutDir2)
			os.mkdir(gfsOutDir2)
		# Download hourly files from NCEP to hour 120.
		for hrDownload in range(1,121):
			fileDownload = "gfs.t" + dCurrent.strftime('%H') + \
						   "z.sfluxgrbf" + str(hrDownload).zfill(3) + \
						   ".grib2"
			url = httpDownloadDir + "/" + fileDownload
			outFile = gfsOutDir2 + "/" + fileDownload
			if not os.path.isfile(outFile):
				try:
					print('Pulling GFS file: ' + fileDownload + ' from NOMADS')
					request.urlretrieve(url,outFile)
				except:
					print("Unable to retrieve: " + url)
					print("Data may not be available yet...")
					continue

		# Download 3-hour files from hour 120 to hour 240.
		for hrDownload in range(123,243,3):
			fileDownload = "gfs.t" + dCurrent.strftime('%H') + \
						   "z.sfluxgrbf" + str(hrDownload).zfill(3) + \
						   ".grib2"
			url = httpDownloadDir + "/" + fileDownload
			outFile = gfsOutDir2 + "/" + fileDownload
			if not os.path.isfile(outFile):
				try:
					print('Pulling GFS file: ' + fileDownload + ' from NOMADS')
					request.urlretrieve(url,outFile)
				except:
					print("Unable to retrieve: " + url)
					print("Data may not be available yet...")
					continue

		# Download 12-hour files from hour 240 to hour 384.
		for hrDownload in range(252,396,12):
			fileDownload = "gfs.t" + dCurrent.strftime('%H') + \
						   "z.sfluxgrbf" + str(hrDownload).zfill(3) + \
						   ".grib2"
			url = httpDownloadDir + "/" + fileDownload
			outFile = gfsOutDir2 + "/" + fileDownload
			if not os.path.isfile(outFile):
				try:
					print('Pulling GFS file: ' + fileDownload + ' from NOMADS')
					request.urlretrieve(url,outFile)
				except:
					print("Unable to retrieve: " + url)
					print("Data may not be available yet...")
					continue

# Remove the LOCK file.
os.remove(lockFile)
