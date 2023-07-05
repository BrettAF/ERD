



import pandas as pd
import numpy as np
import csv
import mne
import regex as re
from mne.io import Raw
from mne import read_events
import os

#These channels are consistent across all the subjects. They are the only ones I Plan to use
consistent_channels=["Time","# FP1-F7","FP1-F7"," F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3"," P3-O1", "FP2-F4","F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "T8-P8-0","P8-O2", "FZ-CZ","CZ-PZ","Label","EEG Time"]

def summaryReader(folder):
  date_time = [0,0,0]
  eeg_files =[]
  file_start_times =[]
  seizure_start_times = [0]
  seizure_end_times = [0]
  print(folder)
  os.chdir(folder)
  summary_name = folder+'-summary.txt'
  print(summary_name)
  summaryfile=open(summary_name,"r")
  for line in summaryfile:

    #Looks for files that need to be opened
    fn_search=re.search("\S+.edf",line)
    if fn_search:
      toopen=line[(fn_search.span()[0]):(fn_search.span()[1]-4)]
      eeg_files.append(toopen)

    #Looks the time files started at
    st_search=re.search("File Start Time: \S+",line)
    if st_search:
      starttime=line[(st_search.span()[0]+17):(st_search.span()[1])]
      previous_time = int(date_time[0])
      date_time = starttime.split(':')


      #Does conversions to seconds
      date_time[0]=int(date_time[0])

      #This while loop wraps midnight arround to keep numbers consistent
      while date_time[0] < previous_time:
        date_time[0] =  date_time[0] + 24

      hour_seconds    = int(date_time[0]) * 60 * 60
      minute_seconds  = int(date_time[1]) * 60
      seconds_seconds = int(date_time[2])

      #This lists the time since the EEG began recording in seconds
      time_since_recording_began = hour_seconds+minute_seconds+seconds_seconds
      file_start_times.append(time_since_recording_began)

    #This lists the time the seizures began
    ss_search=re.search("Seizure Start Time: \S+",line)
    if ss_search:
      seizure_start_times.append(time_since_recording_began + int(line[(ss_search.span()[0]+20):(ss_search.span()[1])]))
    #This searches for seizure end
    se_search = re.search("Seizure End Time: \S+",line)
    if se_search:
      seizure_end_times.append(time_since_recording_began + int(line[(se_search.span()[0]+18):(se_search.span()[1])]))
  print(eeg_files)
  print(file_start_times)

  return   eeg_files, file_start_times, seizure_start_times, seizure_end_times

# This function assigns a label to the data based on the time until the next seizure
def assignLabel(time,seizure_start_times,seizure_end_times ):
  for i in range(0,len(seizure_start_times)):
    #if it is before a seizure
    if (time >=seizure_end_times[i-1]) and (time<=seizure_start_times[i]):
      return seizure_start_times[i]-time
    #if it is during a seizure
    elif(time>seizure_start_times[i]) and (time <seizure_end_times[i]):
      return seizure_start_times[i]-time

  #if it is at the end of the file
  return 0

def convertToCsv(filename):
#filename = 'chb01_01'
  raw = mne.io.read_raw_edf(filename+'.edf', preload=True)
  header = ','.join(raw.ch_names)
  raw.filter( l_freq=0.2, h_freq=None)
#raw.plot_psd(fmax=100)
#fig = raw.compute_psd(fmax=128).plot(average=True, picks="data", exclude="bads")
#add_arrows(fig.axes[1])
  np.savetxt(filename+'.csv', raw.get_data().T, delimiter=',', header=header)

def column_checker(header):
  returnvector=[]
  channels = []
  for i in range(0,len(header)):
      for j in range(0,len(consistent_channels)):
          if header[i]==consistent_channels[j] and header[i] not in channels:
            channels.append(header[i])

            returnvector.append(int(j))
  return returnvector



print(column_checker(consistent_channels))

#This applies lables to a new csv

def convertFile(file_name, recording_start_time):
  with open(file_name+'.csv','r') as csvinput:
    output_name = str(file_name )+ " formated.csv"
    print(output_name)
    with open(output_name, 'w') as csvoutput:
      writer = csv.writer(csvoutput, lineterminator='\n')
      reader = csv.reader(csvinput)


      row = next(reader)
      columns_to_keep=column_checker(row)

      header = []
      for column in columns_to_keep:
        header.append( row[column])
      header.append('EEG Time')
      header.append('Label')
      print(header)
      writer.writerow(header)
      index=0
      for row in reader:
          index=index+1
          row_to_keep=[]
          for column in columns_to_keep:
              row_to_keep.append(row[column])
          row_to_keep.append( index*.003906+recording_start_time )
          row_to_keep.append( assignLabel( index*.003906 + recording_start_time, seizure_start_times, seizure_end_times )  )
          writer.writerow(row_to_keep)

#for i in range(0,len(eeg_files)):
  #print(eeg_files[i])
  #convertToCsv(str(eeg_files[i]))
  #convertFile(eeg_files[i],file_start_times[i])



folders = ['chb05', 'chb06','chb07','chb08','chb09', 'chb10','chb11','chb12','chb13', 'chb14','chb15','chb16','chb17', 'chb18','chb19','chb20','chb21', 'chb22','chb23','chb24',]
for folder in folders:
  eeg_files, file_start_times, seizure_start_times, seizure_end_times = summaryReader(folder)
  print(seizure_start_times)
  print(seizure_end_times)

  for i in range(0,len(eeg_files)-1):
    print(eeg_files[i])
    convertToCsv(str(eeg_files[i]))
    convertFile(eeg_files[i],file_start_times[i])

  os.chdir('..')

#os.chdir('chb01')


#print("Current working directory: {0}".format(os.getcwd() ))
#pd.read_csv('chb01_01 formated.csv').head
#os.chdir('..')
