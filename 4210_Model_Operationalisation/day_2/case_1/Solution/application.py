#Case: Developing Web APIs & Operationalizing Machine Learning Models

#Good luck and have fun!

#First things first - on the bottom of this script we use the app.run function that activates this script
#Make sure to change the port to a number between 5001 - 6000 that other people will not be sharing with you.

#########Case 2: Operationalize your model in production using and API server############

#The case consists of 3 parts:

#    1.Put the model behind an API
#    2.Your first 'web app' using a form
#    3.What's next?

#############Part 1.Put the model behind an API####################
# The basis web server from before, add your new routes to this basis verison!

# Importation
from flask import Flask, request,render_template
import numpy as np
import pandas as pd
import pickle
import logging

#to log to console use: app.logger.info('#########################################')
# Load the model
forest = pickle.load(open('model.pickle', 'rb'))
# Load the column names
col_names = pickle.load(open('columns.pickle', 'rb'))
# Load the imputation mean
impute = pickle.load(open('satisfaction_mean.pickle', 'rb'))
# Load the scaling parameters
scale = pickle.load(open('scaling.pickle', 'rb'))
# Load the salary encoder
encode = pickle.load(open('salary_encoding.pickle', 'rb'))


# A server shutdown function
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
# Initialize the server
application = Flask(__name__)

# Your first API! It is a GET request that simply returns some text.
@application.route('/', methods=['GET'])
def welcome():
    return 'Welcome to your first web server!'

# This is our route that shuts down the server
@application.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


#Now it's time to put your model behind an API. 
#We will start by having the API return a prediction for one person (one datapoint). 
#Next, we will create an API that will make a prediction for all people in the dataset and write this to a new csv.

#We pass the parameters that our model needs through the API into the model. 
#The API will return whether the model predicted that someone would leave or not. 
#Keep in mind that the parameter in the url is a string and you might have to convert it to the data type your model expects.

#What steps should you follow to achieve this in your API route? Have a look below:
#Exercise 1: Create an API route that returns a prediction for one datapoint based on characteristics you pass as parameters into the API

#Steps:

#    1.Extract the parameters from the URL request
#    2. Load the model objects using pickle
#    3.Insert the user input variables into a dictoionary and append it to a DataFrame
#    4.Make the prediction
#    5.Return whether the employee will leave or not based on the predicted value
#
#Tip: Try to write steps 2, 3and 4 seperately before putting them into the API, since the syntax can be confusing.
#This will make debugging much easier! And have a look at the code from part 1 of the case (train the model), that can help you along.

#We can send multiple parameters in the URL like:

#http://127.0.0.1:5000/returnName?name=Charlotte&age=45&hight=179
   
# This function dummifies the department variables using a dictionary
# You can use it 
def dummify_department(value,dict_x):
    if value=='RandD':
        dict_x['RandD']=1
        dict_x['management']=0
        dict_x['other']=0
    elif value=='management':
        dict_x['RandD']=0
        dict_x['management']=1
        dict_x['other']=0
    else: #other provinces
        dict_x['RandD']=0
        dict_x['management']=0
        dict_x['other']=1
    return dict_x

@application.route('/test',methods = ['GET'])
def test():
    dict_x = {}
    variable = 'test'
    dict_x[variable] = (request.args.get(variable))
    return ("welcome " + dict_x[variable] + "!")

# The route that takes the parameters and returns a prediction for one datapoint
@application.route('/predict', methods=['GET'])
def predict():
    dict_x = {} #Open empty dictionary to insert the variables to
    df = pd.DataFrame(columns=col_names) #Open empty DataFrame with exact order of columns as needed for prediction
    
    #Define columns that should be extracted from the API
    #Variables like hours per project are not asked from the user, but calculated in the backend
    variables = ['average_monthly_hours','salary','number_project','last_evaluation','satisfaction_level','department']
    #Use a loop to extract variable by variable and handle each one.
    #this step should make sure of several things:
    #1. average monthly hours, number of projects, satisfaction level (remember: might need to be imputated) and last evaluation should be saved as float
    #2. salary should be kept as string
    #3. department should be dummified using the function above
    for variable in variables: #?average_montly_hours=2&number_project=3&satisfaction_level=0.45&last_evaluation=4&salary=low
        if variable in ('average_monthly_hours','number_project'):
            dict_x[variable]= float(request.args.get(variable))
        elif variable in ('satisfaction_level') and str(request.args.get(variable))=='':
            dict_x[variable]=impute #if satisfaction level is null, fill with the imputated value
        elif variable in ('satisfaction_level','last_evaluation'):
            dict_x[variable] = float(request.args.get(variable))
        elif variable in ('salary'):
            dict_x[variable]=request.args.get(variable) #since we expect a string and not a float, we should add this part
        else: #department
            dict_x = dummify_department(request.args.get(variable),dict_x) #Dummify department variables
    #Create hours per project        
    dict_x['hours_per_project']=dict_x['average_monthly_hours']/dict_x['number_project']  
    #dict_x['average_montly_hours']=0.5
    application.logger.info('#########################################')
    application.logger.info(dict_x)
    application.logger.info('#########################################')
    #Append dictionary to the empty Dataset
    # df = df.append(dict_x, ignore_index=True)
    df = pd.concat([df, pd.DataFrame([dict_x])], ignore_index=True) 
    #Apply the rest of the transformations(salary encoding, monthly hours scaling)
    
    df.replace(encode, inplace=True) #Encode salary
    ## FIX!
    df['average_monthly_hours'] = scale.transform(df['average_monthly_hours'].values.reshape(1, -1)) #Scale monthly hours	
    
	#Predict
    application.logger.info('#########################################')
    application.logger.info(df.loc[0])
    application.logger.info('#########################################')
    prediction = forest.predict(np.array(df).reshape(1,-1))[0]
    if prediction == 0:
        return ("Prediction: Employee won't leave")
    else:
        return ("Prediction: Employee will leave")


#Exercise 2: Create an API route that will make a prediction on the full dataset and writes this to a new CSV

#Next we will make a prediction on all datapoints and write (together with the data one each person) to a new CSV. 
#Write a new route that will take no parameters, but will do a prediction on the full dataset. Therefore the route should:

#this is a simple example of model operationalization: we can initiate a quick process where a user can send some datapoints,
#and recieve the model output in return.
        
#in a (slightly) more complex version of this exercise,
#the user would upload it's own datapoints and recieve a new csv with predictions to download from the server.

#Steps:
#
#    1.Import the data from the csv (HR_new_recruitment.csv)
#    2.Load the model objects and pipeline and make a prediction on the data
#    3.Append these predicitons as a new column to the dataframe that has all the data
#    4.Write this dataframe to a new csv file (Pandas dataframes have a method to write the dataframe to csv!)
#    5.Return on the API route whether the prediction process went successfully or not

def test_transformation(prediction_set):
    prediction_set_copy = prediction_set.copy()
    prediction_set_copy['department'] = prediction_set_copy['department'].apply(lambda x: 'other' if x not in ['RandD','management'] else x)
    prediction_set_copy['last_evaluation'] = prediction_set.last_evaluation.str.replace('%','').astype(int)*0.01
    prediction_set_copy = pd.concat([prediction_set_copy, pd.get_dummies(prediction_set_copy['department'])], axis=1)
    prediction_set_copy.drop('department', axis=1, inplace=True)
    prediction_set_copy.replace(encode, inplace=True)
    prediction_set_copy['hours_per_project'] = prediction_set_copy['average_montly_hours']/prediction_set_copy['number_project']
    prediction_set_copy['average_montly_hours'] = scale.transform(prediction_set_copy['average_montly_hours'].values.reshape(-1,1))
    prediction_set_copy.satisfaction_level.fillna(impute,inplace=True)
    prediction_set_copy = prediction_set_copy.loc[:,col_names]
    return prediction_set_copy
    

# The route that makes a prediction on all data and writes this to a new CSV
# We use a try/except to here to catch any errors during the API call and still return something when an error happens
# Without this try/except the server can just crash when something goes wrong without returning valuable info

#@application.route('/predict_all', methods=['GET'])
#def predict_all():
#    #Hint: you can use the test_transformation function you have made yesterday!
#    try: 
#        df = pd.read_csv(r"C:\Users", header=0, sep=",")
#        X_predict = test_transformation(df)
#    except:
#        return 'Error occured'       
#
#    try:        
#        # Make new predictions
#        predicted = forest.predict(X_predict)
#        
#        # save the predictions to a csv file
#        X_predict['correct_prediction'] = predicted
#        X_predict.to_csv('correct_predictions.csv', header=True)
#        return 'prediction was successful'
#    except:
#        logging.exception('')
    
    
#################Part 2. Your first 'web app' using a form###############
#It is now time to take things one step further. We will use an index.html file so we have a 'front-end' and very basic web application. 
#The index.html page will display a form where characteristics about someone can be entered (instead of in the URL) 
#and a prediction will be returned by the model. In order to do this we will use a POST request instead of a GET request.

#First check out the /index route to see how the form will look like. 
#When you press 'estimate if employee will leave' these will be send to the route predict_post. Can you make sure the correct prediction is returned?

# /Index route that displays the form
@application.route('/index')
def index():
    return (render_template('index.html')) #a form that send a post request to /predict_post


# Define a route that receives a post request and returns the form together with a prediction
@application.route('/predict_post', methods=['POST'])
def pred_post():
        user_input=request.form #we have extracted a dictionary with all variables:values pairs
        df = pd.DataFrame(columns=col_names) #Open empty DataFrame with exact order of columns
        dict_x = {}
        
        #Apply a similar loop as before
        #Think - what has changed from the input we recieved from the URL?
        #Are we expecting to get the same format for the different variables?
        for variable in user_input:
            if variable in ('average_monthly_hours','number_project'):
                dict_x[variable]=float(user_input[variable])
            elif variable in ('satisfaction_level') and user_input[variable]=='':
                dict_x[variable]=impute #if satisfaction level is null, fill with the imputated value
            elif variable in ('satisfaction_level','last_evaluation'):
                dict_x[variable]=float(user_input[variable])/100
            elif variable in ('salary'):
                dict_x[variable]=user_input[variable] #since we expect a string and not a float, we should add this part
            else:
                dict_x = dummify_department(user_input[variable],dict_x) #Dummify department variables
        dict_x['hours_per_project']=dict_x['average_monthly_hours']/dict_x['number_project']    #Create new variable
        
        #Append the dictionary
        df = pd.concat([df, pd.DataFrame([dict_x])], ignore_index=True)

        #Apply the rest of the transformations 
        df.replace(encode, inplace=True) #Encode salary
        df['average_monthly_hours'] = scale.transform(df['average_monthly_hours'].values.reshape(1, -1)) #Scale monthly hours

        #Predict
        prediction = forest.predict(np.array(df).reshape(1,-1))[0]
        prediction_text = 'Employee will leave' if prediction==1 else "Employee won't leave"
        return (render_template('index.html', pred=prediction_text))
    

	
## AWS sample code
def say_hello(username = "World"):
    return '<p>Hello %s!</p>\n' % username
# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>EB Flask Test</title> </head>\n<body>'''
instructions = '''
    <p><em>Hint</em>: This is a RESTful web service! Append a username
    to the URL (for example: <code>/Thelonious</code>) to say hello to
    someone specific.</p>\n'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'

application.add_url_rule('/', 'index2', (lambda: header_text + say_hello() + instructions + footer_text))

# add a rule when the page is accessed with a name appended to the site
# URL.
application.add_url_rule('/<username>', 'hello', (lambda username:header_text + say_hello(username) + home_link + footer_text))
########part 3: What's next?#########   
    
#Call the API directly from python

#We can also call the API directly from python without opening the link to the server in our browser. 
#Please open the py file called case_request for this exercise.

#Run your server in this notebook and open the new notebook case_request.py and do the exercise in that notebook.



if __name__ == '__main__': #move this condition to the end of your script every time you add a new part
    application.run(port=5008,
            debug=True) 