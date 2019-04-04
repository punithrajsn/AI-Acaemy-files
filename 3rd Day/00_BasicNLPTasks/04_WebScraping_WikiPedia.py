# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:11:54 2019

@author: vsurampu
"""

# importing libraries
from bs4 import BeautifulSoup
from urllib import urlopen
import urllib.request
import re

url = "http://en.wikipedia.org/wiki/Artificial_intelligence"

page = urllib.request.urlopen(url) # conntect to website

try:
    page = urllib.request.urlopen(url)
except:
    print("An error occured.")
    
soup = BeautifulSoup(page, 'html.parser')
print(soup)

#-- Identify the regular expression for table of contents --#
regex = re.compile('^tocsection-')
content_lis = soup.find_all('li', attrs={'class': regex})
print(content_lis)

#-- Get an array of list items --#
content = []
for li in content_lis:
    content.append(li.getText().split('\n')[0])
print(content)

#--To get the contents of see also section --#

see_also_section = soup.find('div', attrs={'class': 'div-col columns column-width'})
see_also_soup =  see_also_section.find_all('li')
print(see_also_soup)

#-- To extract the links --#

see_also = []
for li in see_also_soup:
    a_tag = li.find('a', href=True, attrs={'title':True, 'class':False}) # find a tags that have a title and a class
    href = a_tag['href'] # get the href attribute
    text = a_tag.getText() # get the text
    see_also.append([href, text]) # append to array
print(see_also)

#--Saving Data--#
with open('content.txt', 'w') as f:
    for i in content:
        f.write(i+"\n")
        
with open('see_also.csv', 'w') as f:
    for i in see_also:
        f.write(",".join(i)+"\n")
