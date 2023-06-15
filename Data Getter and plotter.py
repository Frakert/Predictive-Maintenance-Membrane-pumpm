# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:40:27 2023
 
aanpassing

@author: klabbf
"""
import warnings
warnings.simplefilter(action='ignore') #pandas read_sql keeps throwing warnings about using SQLAlchemy, this line ingores those errors

import pandas as pd
import pyodbc
import numpy as np
from datetime import datetime, timedelta

#%% Settings

#Get AspenTech instaled for drivers, then define a user Data Source Name (DSN). In this case called freek. 
ODBC_string="DSN=Freek;Driver={AspenTech SQLplus}"

#Define the NAME LIKE uit de Namenlijst query, this is also used for the File name
Name_Like='%5IAL_3_%301%'


"""
#Predefined standard dataset length
start='30-MAR-23 00:01'
end= '30-MAR-23 23:59'

datetime_start = datetime.strptime(start, '%d-%b-%y %H:%M')
datetime_end = datetime.strptime(end, '%d-%b-%y %H:%M')
"""

datetime_start=datetime.now().replace(second=0, microsecond=0)
datetime_end=datetime_start-timedelta(days=1)

start=datetime_start.strftime("%d-%b-%y %H:%M").upper()
end=datetime_end.strftime("%d-%b-%y %H:%M").upper()



filename= 'Data_' + Name_Like + '_from_' + datetime_start.strftime("%Y-%m-%d_%H_%M") + '_until_' + datetime_end.strftime("%Y-%m-%d_%H_%M") 
filepath='C:/Users/klabbf/OneDrive - Canon Production Printing Netherlands B.V/Documents/Data-Excel/Python scripting/%s.csv'%(filename)

#%% SQL Connection and name Query

conn = pyodbc.connect(ODBC_string) 
cur = conn.cursor()

#Grabs all the names from the database that contain 5IAL_3_ (meaning latex plant) and Name_Like, like earlier defined.
name_query="SELECT NAME AS NAME_LIST FROM HISTORY WHERE NAME LIKE '%s';"% (Name_Like) 

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
           'FROM HISTORY(80)'\
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

#%% Plot gotten data
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(filepath,parse_dates=[1])

flow_norm=(data['5IAL_3_FIT301.61MF']-40.062658)/60.937859
pres_norm=(data['5IAL_3_P301.70']-0.486876)/0.859226
pres_pump_norm=(data['5IAL_3_PIT 301.55']-0.19296170517190583)/0.38171527018994034

data['frac']=(pres_norm/flow_norm)

# =============================================================================
# 
# plt.plot(data['0'],data['frac'])
# plt.tick_params(axis='x', labelrotation=90)
# plt.ylim([0,2])
# plt.title('Data Fraction')
# plt.figure()
# 
# =============================================================================
data['frac']= (abs(data['frac']) < 3) * data['frac']

#data['frac']= (norm['frac'] > 0) * norm['frac']

                                             

data['frac'] = data['frac'].rolling(60*6).mean() 



# =============================================================================
# plt.plot(data['0'],data['frac'])
# plt.tick_params(axis='x', labelrotation=90)
# plt.title('6 hours rolling mean frac')
# plt.figure()
# =============================================================================



plt.plot(data['0'],pres_norm,data['0'],flow_norm,data['0'],pres_pump_norm,'r')
plt.tick_params(axis='x', labelrotation=90)
plt.ylabel('Normalised Data')
plt.xlabel('Date and Time')
plt.title('Normalised press and flow data')
plt.legend(['Press Norm','Flow norm'])
plt.ylim([0,2.3])
plt.figure()



plt.plot(data['0'],pres_norm,data['0'],flow_norm)
plt.tick_params(axis='x', labelrotation=90)
plt.ylabel('Normalised Data')
plt.xlabel('Date and Time')
plt.title('Normalised press and flow data')
plt.legend(['Press Norm','Flow norm'])
plt.ylim([0,2.3])
plt.figure()




plt.plot(data['0'],data['5IAL_3_FIT301.61MF'])
plt.tick_params(axis='x', labelrotation=90)
plt.title('Raw flow')
plt.figure()



#%%
import seaborn as sns

plt.plot(data['0'],data['5IAL_3_P301.70'])
plt.tick_params(axis='x', labelrotation=90)
sns.lineplot(data['0'],data['5IAL_3_PIT 301.55'],hue=data['5IAL_3_301.BatchName'])
plt.figure()

plt.plot(data['0'],data['5IAL_3_FIT301.61D'])

