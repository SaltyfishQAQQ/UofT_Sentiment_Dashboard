from twikit import Client, TooManyRequests #interact with twitter; rate limit errors
import time
from datetime import datetime
import csv
from configparser import ConfigParser
# Read the config file
from random import randint

MINMUM_TWEETS = 1000
Query = 'chatgpt'

# login credentials
config = ConfigParser()
config.read('config.ini')
username = config['X']['username']
email = config['X']['email']
password = config['X']['password']
# 需要重置密码后连接或用新账户

# use credentials -》use cookies
client = Client(language='en-US')
client.login(auth_info_1=username, auth_info_2=email, password=password)
client.save_cookies('cookies.json')
