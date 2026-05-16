"""
Case: Model Operationalization
Part 2: Build your first web server and APIS

Good luck and have fun!
---------------------------


Now it's time to build your first web server! On the web server we can create our API routes.

The server can be shut down by CTRL + C in the CMD.

More about Flask can be found at the links below. We also stimulate you to use Google and Stack
Overflow if you get stuck! Or ask for help :)

Flask documentation: https://flask.palletsprojects.com

Run this script to spin up the server (run python case_part_2.py in a terminal)
Test if the server works by going to http://localhost:5008
See if it returns 'Welcome to your first web server!' which the main route should return.


Then it is time to add some more API routes to our web server!

Now we will create some new routes and add these to our web server. Please go through the following
exercises.  You can use the code blok above to add the routes. New routes can be added under the
line: # NEW API ROUTES FROM THE EXERCISES CAN BE ADDED BELOW THIS LINE. Be sure that the if
__name__ == '__main__': part is at the end of the script.


EXERCISES
---------------------------------------------------------------------------

1. Let's add an API route
Create a new route similar to the index route ('/') the server now has.
This route should receive the request on '/hello'.
It should be a GET request that returns the message 'Hello'.
Visit the url in your browser and test if the route works correctly.

2. Time to show a GIF
Create a route that will return the gif 'ok.gif' and show the gif in the browser.
You can find the gif file in the same directory as this notebook.
Use Google and the Flask documentation to find out how to do this!

3. Working with parameters
Let's start working with parameters!
As you remember from the training we can send parameters with our API request. Make sure that your
API can extract the parameter that is given with the request and return the parameter. Create a
route called welcomeName that will return the text 'Welcome (name passed as parameter here)!'

4. Puzzle time
Let's combine our learnings!
Let's assume that you have a built a puzzle and your API will check the solution of the puzzle. You
are free to decide what the solution should be. The API will simply take a parameter 'answer' and
return a gif. Which gif will depend on whether the solution was correct or not. Keep in mind that
the answer parameter will be a string.  If the answer is correct return 'winner.gif', otherwise
return 'loser.gif'.

As an extension to this exercise, you might want to return an explanation of the puzzle if the
`answer` parameter has not been provided at all, and if there is an `answer` parameter, check if it
is correct.

"""

from flask import Flask


# This code sample will spin-up your first web server!
# Run the code and then visit the web server at http://localhost:5008 or http://127.0.0.1:5008
# If multiple users try to run the app on the same port you might get a different port number
# In case of issues with the port number, you can change it like this: app.run(port=5001)


# Initialize the Flask server
# [[ not required to understand how to initialize Flask: ]]
#   The __name__ variable is a special variable in Python that refers to the filename
#   of the current file (e.g. 'Case_part_2.py'). However the file that is referred to
#   when running python (remember `python file.py` refers to 'file.py'), has a __name__
#   variable equal to "__main__".
app = Flask(__name__)


# Your first API! It is a GET request that simply returns some text.
@app.route("/", methods=["GET"])
def welcome():
    return "Welcome to your first web server!"


# NEW API ROUTES FROM THE EXERCISES CAN BE ADDED BELOW THIS LINE

# 1. Add a new API route that returns hello
# ...


# 2. Add a new API route that returns a gif, using the send_file function from Flask
# ...


# 3. Add a new API route that uses parameters and returns Welcome + the name you entered
#
# Remember: when creating an API route (e.g. '/welcome'), in your browser you can always go to
# '/welcome?key=value' where both 'key' and 'value' can be arbitrarily chosen. (e.g.
# /welcome?country=NL).
# These parameters are called 'URL parameters', find out in the docs how to extract this parameter
# from your request.
# ...


# 4. Add a new API route that checks the answer to a puzzle and returns a gif depending on whether
# it was correct or not. You can use the provided winner/loser gifs
# ...


# These two lines should always be at the bottom of the script. Remember when we said that `python
# file.py` changes the __name__ variable to '__main__' for the file 'file.py'. That means that in
# this special case, where this is the file we started (and not in all other files), we want to
# start a Flask server.
if __name__ == "__main__":
    app.run(port=5008, debug=True)  # Instantly see changes on the server
