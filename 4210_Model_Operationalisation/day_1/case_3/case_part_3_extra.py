"""
Case: Model Operationalization
Part 3 extra: Calling your own API from python

Good luck and have fun!
---------------------------

Small extra exercise: if you finish early, use the requests package from yesterday to directly call your model API from python, without opening the link to the server in our browser
"""

import requests

# Fill in the gaps below:

# The API endpoint
endpoint = ""

# The request parameters (in a Python dictionary)
params = {}

# Make the request
r = requests.get(url=endpoint, params=params)

# Extract the response
response_text = r.text
print(response_text)
