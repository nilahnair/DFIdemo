import json
import platform
import os
import io
import logging
import datetime
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
    
    user, pw, url, db_name = load_credentials(path='~/.mongodb_credentials')

    return MongoObserver.create(url=url,
                                         db_name=db_name,
                                         username=user,
                                         password=pw,
                                         authSource='admin',
                                         authMechanism='SCRAM-SHA-1')
    
def load_credentials(path='~/.mongodb_credentials'):
    path = os.path.expanduser(path)
 
    logger = logging.getLogger('::load_credentials')
    logger.info(f'Loading credientials from {path}')
    with io.open(path) as f:
        user, pw, url, db_name = f.read().strip().split(',')
 
    return user, pw, url, db_name