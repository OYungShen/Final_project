#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import request, Flask,jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from ResModel import RES
import json
from io import BytesIO
from base64 import b64decode


# In[ ]:


app = Flask(__name__)

@app.route("/Captcha", methods=['POST'])
def get_frame():
    device = torch.device( "cpu")
    res_CNN = RES().to(device)
    res_CNN.eval()
    res_CNN.load_state_dict(torch.load('./res_direct.pkl', map_location=torch.device('cpu')))
    res_CNN.to(device)
    LETTERSTR = "2346789abcdefghjklmnpqrtuxyz"

    res = json.loads(request.data.decode('utf-8'))  # 获取推过来的json，也可以用data然后转换成json
    image_data = BytesIO(b64decode(res["Captcha"]))
    img = Image.open(image_data)
    
    vimage = Variable(transforms.ToTensor()(img).unsqueeze(0)).to(device)
    out1,out2,out3,out4,out5 = res_CNN(vimage)

    c0 = LETTERSTR[torch.max(out1, 1)[1]]
    c1 = LETTERSTR[torch.max(out2, 1)[1]]
    c2 = LETTERSTR[torch.max(out3, 1)[1]]
    c3 = LETTERSTR[torch.max(out4, 1)[1]]
    c4 = LETTERSTR[torch.max(out5, 1)[1]]
    c = '%s%s%s%s%s' % (c0, c1, c2, c3, c4)
    
    return jsonify({'Captcha': c})

@app.route("/test", methods=['GET'])
def test():
    return jsonify({'test': "test"})

if __name__ == "__main__":
    app.run(host= '0.0.0.0',port='8080')


# In[ ]:




