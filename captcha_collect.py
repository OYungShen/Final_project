#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from time import sleep
import pandas as pd
import requests
import json
from PIL import Image
from base64 import b64encode,b64decode
from io import BytesIO
from bs4 import BeautifulSoup
import threading
from requests.adapters import HTTPAdapter


# In[ ]:


img_flag = False
token_flag = False
read_time = 20
delay = 0.1
s = requests.session()
s.mount('http://', HTTPAdapter(max_retries=30))
s.mount('https://', HTTPAdapter(max_retries=30))



while token_flag == False:
    try:
        r =s.get("https://youthdream.phdf.org.tw/member/login",timeout=read_time)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            _token = soup.select_one('input[name = "_token"]')['value']
            token_flag = True
        sleep(delay)
    except:
        None
## get cpatcha_url
_ = soup.select_one('a[id = "captcha"]').select_one('img')
captcha_url = _["src"]

for i in range(1000,1003):
    ## get captcha_img   
    img_flag = False
    while img_flag == False:
        try:
            cap_img =s.get(captcha_url,timeout=read_time)
            if cap_img.status_code == 200:
                image = Image.open(BytesIO(cap_img.content))
                image = image.convert('RGB')
                img_flag = True
                image.save('./data/captcha_original/jpg/'+str(i)+'.jpg')
        except:
            None


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#folder = "5_trans_set/"
folder = "5_captc_rel/"

filename_list = []
for i in range(1000,1003):
    print(i)
    im = Image.open('./data/captcha_original/jpg/'+str(i)+'.jpg') 
    im = im.resize((160,60))
    plt.imshow(im)
    plt.show()
    file_name = input()
    im.save('./data/'+folder+file_name+'.jpg')
    plt.clf()


# In[ ]:




