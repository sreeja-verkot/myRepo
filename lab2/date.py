import nltk
import re
from bs4 import BeautifulSoup
from urllib import request
url = "https://gecskp.ac.in/"
response = request.urlopen(url)
html = response.read().decode('utf8')
raw = BeautifulSoup(html).get_text()
mail = re.findall('\S+@\S+', raw)     
print("Email id is :",mail)
date = re.findall('^(0[1-9]|[12][0-9]|3[01])[- /.](0[1-9]|1[012])[- /.](19|20)\d\d$', raw) 
print("Date is :",date)
