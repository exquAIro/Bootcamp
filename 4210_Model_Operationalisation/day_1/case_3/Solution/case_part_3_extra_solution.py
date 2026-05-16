"""
Case: Model Operationalization
Part 3 extra: Calling your own API from python

Good luck and have fun!
---------------------------

Small extra exercise: if you finish early, use the requests package from yesterday to directly call your model API from python, without opening the link to the server in our browser
"""

import requests

# The API endpoint
endpoint = "http://127.0.0.1:5008/predict"

# The request parameters (in a Python dictionary)
params = {
    "average_monthly_hours": 300,
    "number_project": 3,
    "last_evaluation": 0.2,
    "satisfaction_level": 0.2,
    "salary": "low",
    "department": "other",
}

# Make the request
r = requests.get(url=endpoint, params=params)

# Extract the response
response_text = r.text
print(response_text)

# Output should be "Prediction: Employee will leave"
