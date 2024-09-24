import json
from urllib.parse import urlparse
from sacred.observers import MongoObserver

def create_observer(credentials_path='credentials.json'): 
    # Load credentials from credentials.json
    #with open('credentials.json', 'r') as file:
    #    credentials = json.load(file)
    
    # Get the first (and only) value at the top level
    #root_name, values = next(iter(credentials.items()))

    # Extract credentials from the data inside the root
    #mongodb_uri = values.get('mongodbURI', '')
    #path = values.get('path', '')

    # Parse MongoDB URI to extract username, password, and db name
    #parsed_uri = urlparse(mongodb_uri)
    #username = parsed_uri.username
    #password = parsed_uri.password
    #db_name = path.strip('/')

    # Create MongoObserver and append it to ex.observers
    return MongoObserver.create(url='curtiz', db_name='nnair', username='nnair', password='wfe5NjN8', authSource='admin', authMechanism='SCRAM-SHA-1')
    