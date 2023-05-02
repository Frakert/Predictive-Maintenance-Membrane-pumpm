# Predictive Maintenance Membrane pump
Author: Freek Klabbers (klabbf)\
Date: 1-5-2023\
Location: Venlo\
Department: CPP Venlo -> M&L -> M2 -> TS -> ME\
Mentors: Ton Driessen and Marco van Hout\
\
This GitHub was created, hosted and under managment of Freek Klabbers, an gradution intern for CPP from Feb 2023 until Jun 2023.
The assignment was to implement a predictive maintenance system for the latex ink plant 1R in Venlo.
For this code needed to be written, this places serves for its version controll and issue tracking.

## General Design
The general design of the application is shown below:
![image](https://user-images.githubusercontent.com/85930277/235468904-d0a9807c-dd62-4d6d-9db9-06e8b9ebb011.png)
My original goal was to develop a model that could accuratly predict failure. I used Kaggle mostly for this so much of this process is not documented within GitHub.\
Since my ambitions have been expanded to actually deploy the system. For this i decided that version controll would be good to implement, thus this repository.\
I have attempted to retroactivly include as much as i could in one 1 place.\
\
The file structure is as follows:
* **Data Analysis:** includes Kaggle files including output when i was still in the exploratory phase. This code however isnt runnable unless you change the input by letting it read csv files again. Also you will need an envoirment which i have definintly not created (yet). Might fix this in the future with the [kaggle api](https://www.kaggle.com/docs/api).

* **Database connections:** Includes the pieces of code I used to fetch data from the AspenTech database using the ODBC (32-bit) drivers named Aspen SQLplus. This is also includes a guide on how to install it hopefully. Check out the documentation i provided there

* **Model for Deployment:** This folder contains everything the Memebrane_Model Class needs to opperate. It also includes the Membrane_Model class itself. The idea is that if the folder is coppied and the model is called it should work right out of the box.

* **Model Training and Optimisations:** This folder contains all the Kaggle scripts used to try models and shows its peformance (hopefully). These Kaggle files again dont run anymore (for now). Every approach has its own file. The clean aggregating is used to optimise the end model. Here factorial analysis was conducted and validated by k crossfold validation. I also added fetched data from the aspentech database (<100 MB).


## Model
The current model that is being implemented is a XGBoost classification model. It's peformance is pictured below.\
Its target is labled data that is labled 20 Batches before failure.
 ![image](https://user-images.githubusercontent.com/85930277/235467818-5611ed23-8a9b-4149-ab77-e5022c232893.png)

XGBoost means eXtreme Gradient Boosting and is a supervised ensemble learning algorithm. It works by using multiple decision tree's that feed their prediction to eachother. It is an quick and effictive way to do classification and regression tasks. The package used is called py-XGBoost and its documentation can be found [here](https://xgboost.readthedocs.io/en/stable/).

