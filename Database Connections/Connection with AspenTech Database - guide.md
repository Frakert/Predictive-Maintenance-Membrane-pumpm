# Connection met de AspenTech Database
Author: Freek Klabbers (klabbf)\
Date: 2-5-2023\
Location: Venlo\
Department: CPP Venlo -> M&L -> M2 -> TS -> ME\
Mentors: Ton Driessen and Marco van Hout\

## Short introduction
This file was made to document my methods of aquiring the data from the AspenTech database.
This was orignally inteded as a quick way to programaticly get my data without having to fetch it by hand.
It works by using the AspenTech Info21 SQLplus ODBC drivers to connect to the database and make an SQL querry.
See explanation below.\

## Full guide
**Note** This has been done on AspenTech V8.7 which is absurdly outdated and as of writing this was in the process of being replaced by a newer version (V12.2).

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


