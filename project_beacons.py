# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:23:52 2018

@author: Schnitzel
"""

import pandas as pd 
import numpy as np 

# Import data 
data = pd.read_csv('beacons_dataset.csv', sep=';', index_col=False)
#print(data.dtypes,'\n')

# Part 1a - Correct room labels
print("Different values of field 'Room': \n", data['room'].unique(),"\n")   
   
data.replace(['Kitch','Kitcen','K','Kithen','Kitchen2','Kitvhen','kitchen','Kichen','Kiychen','Kitcheb','Pantry'],'Kitchen',inplace=True)     
data.replace( ['Bed','Bedroom2','Bedroom1','bedroom','Bedroom1st','Bedroom-1','Chambre','2ndRoom'],'Bedroom',inplace=True)     
data.replace( ['Barhroom','Bathroon','Baghroom','Bsthroom', 'Bathroom1','Bathroim','Bathroom-1','Bathroom-1','Bqthroom'],'Bathroom',inplace=True)       
data.replace( ['Livroom','Living','Livingroom2','Livingroom1','Sittingroom','Leavingroom','LivingRoom','Luvingroom1','LeavingRoom',
               'LuvingRoom','LivingRoom2','Liningroom','Livingroon','livingroom','LivibgRoom','Leavivinroom','TV','Sitingroom',
               'Sittigroom','SittingRoom','SeatingRoom','SittingOver','Sittinroom'],'Living Room',inplace=True)     
data.replace(['Desk','Workroom','Library','Office1','Office2','Office1st','Office-2'],'Office',inplace=True)
data.replace (['Washroom','Laundry','LaundryRoom'],'Laundry Room',inplace = True)
data.replace (['DiningRoom','DinnerRoom','DinningRoom','Dinerroom','DinerRoom'],'Dining Room' ,inplace = True)
data.replace (['Veranda','Garden'],'Outdoor',inplace = True)
data.replace (['Box','Box-1','Storage'],'Storage',inplace = True)
data.replace (['Hall','ExitHall','Entrance','Entry'],'Hall',inplace = True)
data.replace (['Two','Four','three','T','One','Three','Right','Left'],np.nan,inplace = True)

print("Corrected room labels: \n", data['room'].unique(),"\n")

# Part 1b - Remove erroneous users
data = data[ (data['part_id'].str.isnumeric()) & (data['part_id'].str.len()==4) ]

print (data['part_id'].unique().tolist())

#*******************************************************************************************************
# Part 1c- Generate features

tdata = data.copy()
# Create new column with both date and time values and sort them for each participant
tdata['Date/Time'] = data['ts_date'].map(str) + ' ' + data['ts_time'].map(str)
tdata['Date/Time'] = pd.to_datetime(tdata['Date/Time']) #convert date in year-month-day format
tdata = tdata.sort_values(['part_id','Date/Time']) #sort according to date/time for each participant

# Calculate the time spent in each room
tdata['time_diff'] = tdata['Date/Time'].diff()
tdata.loc[tdata['part_id'] != tdata['part_id'].shift(), 'time_diff'] = None
tdata['time_spent'] = tdata.time_diff.shift(-1)
tdata['time_spent_secs'] = tdata.time_spent.dt.total_seconds()
# Keep entries in which time spent in a room is less than a day
tdata = tdata[tdata['time_spent_secs']<86400]  
tdata.drop(['ts_date','ts_time','time_diff','Date/Time','time_spent'], axis=1, inplace=True)

tdata = tdata.sort_values(['part_id','room'])
# Calculate total time spent in house for each participant
users = tdata.groupby(['part_id']).sum()

# Calculate total time spent in each room for each participant
udata = tdata[tdata['room'].isin(['Bathroom','Kitchen','Bedroom','Living Room'])]
udata = udata.groupby(['part_id', 'room']).sum().sum(level=['part_id', 'room']).unstack('room').fillna(0).reset_index()
udata.columns = ['part_id', 'Bathroom', 'Bedroom', 'Kitchen','Living Room']

gdata = pd.merge(users,udata,on='part_id')

gdata['Bedroom']=gdata['Bedroom']/gdata['time_spent_secs']*100
gdata['Kitchen']=gdata['Kitchen']/gdata['time_spent_secs']*100
gdata['Living Room']=gdata['Living Room']/gdata['time_spent_secs']*100
gdata['Bathroom']=gdata['Bathroom']/gdata['time_spent_secs']*100
# Round percentages to the first decimal
gdata = gdata.round(1)
#gdata['tot_perc'] = gdata['Bathroom'] + gdata['Kitchen'] + gdata['Living Room'] +gdata['Bedroom']
gdata.drop(['time_spent_secs'],axis=1,inplace=True)

# Save final dataset as a csv file
gdata.to_csv('beacons_final.csv',sep=';',index=False)





