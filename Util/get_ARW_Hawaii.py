# Quick and dirty program to pull down operational 
# Hawaii 48-hour ARW forecasts. 

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
import shutil
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

outDir = "/glade/p/cisl/nwc/karsten/NWM_v21_Dev/INPUT/WRF_ARW_Hawaii"
tmpDir = "/glade/scratch/karsten"
lookBackHours = 72 # How many hours to look for data.....
cleanBackHours = 240 # Period between this time and the beginning of the lookback period to cleanout old data.  
lagBackHours = 6 # Wait at least this long back before searching for files. 
dNowUTC = datetime.datetime.utcnow()
dNow = datetime.datetime(dNowUTC.year,dNowUTC.month,dNowUTC.day,dNowUTC.hour)
ncepHTTP = "https://ftp.ncep.noaa.gov/data/nccf/com/hiresw/prod"

# Define communication of issues.
emailAddy = 'jon.doe@youremail.com'
errTitle = 'Error_get_ARW_HI'
warningTitle = 'Warning_get_ARW_HI'

pid = os.getpid()
lockFile = tmpDir + "/GET_ARW_HI.lock"

for hour in range(cleanBackHours,lookBackHours,-1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

	# Go back in time and clean out any old data to conserve disk space. 
	if dCurrent.hour != 0 and dCurrent.hour != 12:
		continue # WRF-ARW Hawaii nest data is only available for 00/12 UTC.. 
	else:
		# Compose path to directory containing data. 
		cleanDir = outDir + "/hiresw." + dCurrent.strftime('%Y%m%d%H')

		# Check to see if directory exists. If it does, remove it. 
		if os.path.isdir(cleanDir):
			print("Removing old data from: " + cleanDir)
			shutil.rmtree(cleanDir)

# Now that cleaning is done, download files within the download window. 
for hour in range(lookBackHours,lagBackHours,-1):
	# Calculate current hour.
	dCurrent = dNow - datetime.timedelta(seconds=3600*hour)

	if dCurrent.hour != 0 and dCurrent.hour != 12:
		continue # WRF-ARW Hawaii nest data is only available for 00/12 UTC... 
	else:
		arwOutDir = outDir + "/hiresw." + dCurrent.strftime('%Y%m%d')
		httpDownloadDir = ncepHTTP + "/hiresw." + dCurrent.strftime('%Y%m%d')
		if not os.path.isdir(arwOutDir):
			os.mkdir(arwOutDir)

		nFcstHrs = 48
		for hrDownload in range(1,nFcstHrs + 1):
			fileDownload = "hiresw.t" + dCurrent.strftime('%H') + \
						   "z.arw_2p5km.f" + str(hrDownload).zfill(2) + \
						   ".hi.grib2"
			url = httpDownloadDir + "/" + fileDownload
			outFile = arwOutDir + "/" + fileDownload
			if os.path.isfile(outFile):
				continue
			try:
				print("Pulling ARW file: " + url)
				request.urlretrieve(url,outFile)
			except:
				print("Unable to retrieve: " + url)
				print("Data may not available yet...")
				continue

# Remove the LOCK file.
os.remove(lockFile)
