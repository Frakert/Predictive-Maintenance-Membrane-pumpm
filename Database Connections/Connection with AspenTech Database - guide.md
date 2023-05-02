# Connection met de AspenTech Database
Author: Freek Klabbers (klabbf)\
Date: 2-5-2023\
Location: Venlo\
Department: CPP Venlo -> M&L -> M2 -> TS -> ME\
Mentors: Ton Driessen and Marco van Hout\

# Short introduction
This file was made to document my methods of aquiring the data from the AspenTech database.
This was orignally inteded as a quick way to programaticly get my data without having to fetch it by hand.
It works by using the AspenTech Info21 SQLplus ODBC drivers to connect to the database and make an SQL querry.
See explanation below.\

# Full guide
**Note** This has been done on AspenTech V8.7 which is absurdly outdated and as of writing this was in the process of being replaced by a newer version (V12.2).

## Configuring ODBC
This method will need the AspenTech ODBC drivers to work, to get these install AspenTech as a whole. To do this make a request at the IT Support portal for AspenTech.\

![image](https://user-images.githubusercontent.com/85930277/235662071-66817b00-cefa-41e9-a0aa-594aef43ebcd.png)


After going through the instalation procces with the IT staff do the following:
Press the windows key and search: *"ODBC"*

The following program should pop up: *ODBC Data Sources (32-bit)*: (make sure you select 32-bit and not 64-bit)

![image](https://user-images.githubusercontent.com/85930277/235662833-a9e14e50-8844-486a-b581-1b2dc145bb55.png)

\

When you open the program the following screen should appear, on this screen press "add":  

![image](https://user-images.githubusercontent.com/85930277/235663953-35967322-e277-4460-8361-c0495aefe76d.png)  
\
If the instalation went correctly *"AspenTech SQLPlus"* is one of the drivers that should show up. In the image below it got selected in blue.


![image](https://user-images.githubusercontent.com/85930277/235664143-47690337-657b-48c8-aa4a-9f7bef5b25cf.png)

If configured correctly CP-WAS511 should appear under the Aspen Data Source. Give this connection a name under *'ODBC Data Source Name'*, this Data Source Name is called DSN and will be required later when referercing this connection. Press Ok to close this window.

![image](https://user-images.githubusercontent.com/85930277/235687147-25bc0a32-63eb-48e6-acfb-2f03d7f58fbe.png)

Your new connection should now be in the list under *'User Datasource Names'*. If this is all done you may close the ODBC Data Source Administrator.


## Using ODBC - Excel
The ODBC connection you have just created is very universal and can thus be used by a lot of programs.
If desired (though i dont recomend it) you could use Excel to get data this way too.
Openup a new Excel file, navigate to the Data tab. Under the Data tab find *"Get Data"*, then select *"From other sources"* and finally *"from ODBC"*.
See image below what this should look like.

![image](https://user-images.githubusercontent.com/85930277/235688600-8a405b8d-1087-44bf-9408-dba4d5197245.png)

A window might appear asking for credentials, just fill in your own credentials and it will be fine.
From here a window will appear asking for a DSN, select the DSN configured above. You can also make SQL querries here, which is what i would recomend. If you however dont make an SQL query a navigator will appear. Here you can select whatever you may need. Below is a simple SQL querry used in excel.
![image](https://user-images.githubusercontent.com/85930277/235692685-039af420-9bf9-44f2-9664-eebdb742a6c9.png)


## Using ODBC - Python
Using python has 3 advantages over Excel.
1. I have already done most of the heavy lifting, so you dont have to. I barely used Excel and that is for the following 2 reasons.
2. The History table (where all data is stored) has the sensor names stored in a single column, so what is NOT the case is that every sensor name has its own column. This limits Excel in 2 ways. First, Excel can only handle 1 milion lines. So the amount of data you can fetch at once is very limited. Second if you fetch data this way you are going to have to sort it yourself which is a lot of work.
3. Excel only allows a single SQL querry at a time and does not allow for programaticly querrying. This means that work once again needs to be done by hand, which i do not support.

Now you know why you should want python, now its time for how. I will start with a short Python instalation guide, if you already have python installed or know what you are doing you can skip ahead.\

The easiest way to install Python is probably by using Anaconda, the Anaconda installer can be found [here](https://www.anaconda.com/download/).
**Please be Sure to install 32-bit Anacona!**
Simply follow the steps and install anaconda. For our uses this is probably not neccecary but [CPP does support Anaconda and has a licence for it.](https://wiki.cppintra.net/confluence/display/03173/Python)
Anaconda will install with a lot of base packages, including all packaged that should be neccacary.\

After instalation open Anaconda Navigator, here you can choose your favourite IDE (Integrated Development Envoirment). If you do not know what IDE to choose i recomend Spyder, for the rest of the tutorial i will be using spyder as well.

![image](https://user-images.githubusercontent.com/85930277/235695633-583dcde6-ef95-4d36-ae6a-42692391b50c.png)

Make sure you have *Data Getter.py* downloaded from the GitHub Repository and saved on your local device.
Now open Spyder and in Spyder open the Data Getter.py file.
This should now look something like this:

![image](https://user-images.githubusercontent.com/85930277/235696975-d8eb917d-aed8-4509-9370-5deb135bd06d.png)

In the file close to the top just under the import section there should be section called settings. This section was made in such a way that you wont break other code (as easily) when changing certain properties. The properties you can change have been documented with comments but i have also put them below with red tekst.

![image](https://user-images.githubusercontent.com/85930277/235698495-545b47d0-4ee3-4847-b5e5-b40b7cc5b3a9.png)

You should at the very least change the DSN (if you choose a different name then i did) and the filepath before you run the code. Currently its configured to get data from *'%5IAL%301%'*, which means it will get all data fields that have the subtekst 5IAL and 301 in it. 5IAL is the latex ink 1R ink production, 301 is the mixing vessel. Thus this querry will automaticly get me all the data fields from the mixing vessel between start and end date. It will get stored at the filepath location. 
\
If after changing you press the *run* button (the green play button or just press f5), the program will start to get all the data. In the console it will update you on how much it still has left to go. When its done it will give a message like *"Importing Data all done!"*.
If you then navigate to your specefied filepath you will find a .csv that contains the data you querried:
![image](https://user-images.githubusercontent.com/85930277/235700566-58abfa79-6c33-4af5-bc67-641d52957134.png)

