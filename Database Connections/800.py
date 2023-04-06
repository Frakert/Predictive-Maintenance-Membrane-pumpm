"""
Made by: Freek Klabbers - CPP intern 2023
Department: M2 - Technical Services - Maintenance Engineering
Mentors: Ton Driessen and Marco van Hout

This script was made to automaticly get all the data from AspenTech database.
I made this to first get all names that match the 'Like' clause with a SQL querries.
Then the program will get every datapoint from a sensor name in the defined time period.
Every sensor will get its own column, following the tidy data methology, this column gets written to a csv file.

If less famaliar with programming i would recomnend you only change lines under the 'settings' tab.
At the very least ODBC_string and filepath will need to be changed if used by a new user.

Keep in mind that i used AspenTech ODBC drivers for this, open in windows ODBC Data Sources to view these drivers.
Also this script only works for Python 32-bit, becuase the pyodbc package can only acces its own bit version of drivers.
"""
#%% Imports

import warnings
warnings.simplefilter(action='ignore')

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pyodbc

#%% Settings

#pyodbc connection string, see its documentation for help.
#Get AspenTech instaled for drivers, then define a user Data Source Name (DSN). In this case called freek. 
ODBC_string="DSN=Freek;Driver={AspenTech SQLplus}"


#Define the NAME LIKE from the name getting querry. This case 5IAL is latex, 301 is mixing vessel.
Name_Like='%P800%'

#Define data start and end times here and only here.
start='10-APR-22 12:00'
end='4-APR-23 23:59'

datetime_start = datetime.strptime(start, '%d-%b-%y %H:%M') # ignore this
datetime_end = datetime.strptime(end, '%d-%b-%y %H:%M') # ignore this


#Define the name under which the data is stored
filename= 'Data_' + Name_Like + '_from_' + datetime_start.strftime("%Y-%m-%d_%H_%M") + '_until_' + datetime_end.strftime("%Y-%m-%d_%H_%M")
 
#Define the folder where the data is stored, %s.csv inserts the filename string.
filepath='C:/Users/klabbf/OneDrive - Canon Production Printing Netherlands B.V/Documents/Data-Excel/Python scripting/%s.csv'%(filename)

#%% SQL Connection and name Query

conn = pyodbc.connect(ODBC_string) 
cur = conn.cursor()

#Grabs all the names from the database that contain 5IAL_3_ (meaning latex plant) and Name_Like, like earlier defined.
name_query="SELECT NAME AS NAME_LIST FROM HISTORY WHERE NAME LIKE '%s';"% (Name_Like) # This is an SQL query

names=pd.read_sql(name_query,conn)

#%% Date time

#Create data array, this way dont have to query time 
date_range = np.arange(datetime_start, datetime_end, timedelta(minutes=1)).astype(datetime)

df=pd.DataFrame(date_range)

#%% Get sensorname, make its own column

for i in range(len(names)):
    
    print('Importing ' + names['NAME_LIST'][i])
    tag = names['NAME_LIST'][i]
    
    query= 'SELECT VALUE AS "%s"'\
           'FROM HISTORY(10)'\
           "WHERE (TS BETWEEN '%s' AND '%s') AND NAME='%s';"% (tag, start, end, tag)
          
    data=pd.read_sql(query,conn)

    df=pd.concat([df,data],axis=1)
    
    progress='{} out of {} imported'.format(i+1,len(names))
    print(progress)
    print(' ')
    
#%% Export to Excel cell
df.to_csv(filepath)
print('Exporting Data all done!')

#%% Close connection, possible memory leak
conn.close()
#%%
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(filepath,parse_dates=[1])

plt.plot(data['0'],data['5IAFL_3_P800-PIT403.50'].rolling(60*24*7).mean())
plt.tick_params(axis='x', labelrotation=45)