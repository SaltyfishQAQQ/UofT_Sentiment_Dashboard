from twikit import Client, TooManyRequests #interact with twitter; rate limit errors
import time
from datetime import datetime
import csv
from configparser import ConfigParser
# Read the config file
from random import randint
import os

MINMUM_TWEETS = 1000
Query = 'chatgpt'

# login credentials
config = ConfigParser()
config.read('config.ini')
username = config['X']['username']
email = config['X']['email']
password = config['X']['password']
print("Config file loaded successfully")


# use credentials -》use cookies
client = Client(language='en-US')

# Check if cookies file exists first
if os.path.exists('cookies.json'):
    print("Loading existing cookies...")
    client.load_cookies('cookies.json')
else:
    print("No cookies found, logging in...")
    client.login(auth_info_1=username, auth_info_2=email, password=password)
    client.save_cookies('cookies.json')
    print("Login successful, cookies saved")
        
print(f"Authentication error: {e}")
print("Check your credentials in config.ini or try resetting your password")
exit(1)