# -*- coding: UTF-8 -*-
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
import csv
import numpy as np

class mydataset(Dataset):

    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform
        # print(self.train_image_file_paths[0].split(os.path.sep)[-1])

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        
        label1 = toonehot(image_name[0]) # 为了方便，在生成图片的时候，图片文件的命名格式 "4个数字或者数字_时间戳.PNG", 4个字母或者即是图片的验证码的值，字母大写,同时对该值做 one-hot 处理
        label2 = toonehot(image_name[1])
        label3 = toonehot(image_name[2])
        label4 = toonehot(image_name[3])
        label5 = toonehot(image_name[4])
        label = {
            'label1':label1,
            'label2':label2,
            'label3':label3,
            'label4':label4,
            'label5':label5
        }
        return image, label

transform = transforms.Compose([
    # transforms.ColorJitter(),
    #transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


LETTERSTR = "2346789abcdefghjklmnpqrtuxyz"
def toonehot(text):
    labellist = []
    for letter in text:
        onehot = np.zeros(28, dtype=float)
        #onehot = [0 for _ in range(36)]
        num = LETTERSTR.find(letter)
        onehot[num] = 1.0
    return onehot

def get_train_data_loader():

    dataset = mydataset("./data/5_train_set/", transform=transform)
    return DataLoader(dataset, batch_size=80, shuffle=True)


def get_predict_data_loader():
    dataset = mydataset('./data/5_captc_rel/', transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)

def get_transfer_data_loader():
    dataset = mydataset('./data/5_trans_set/', transform=transform)
    return DataLoader(dataset, batch_size=8, shuffle=True)

# import pytorch_dataset
# predict_dataloader = pytorch_dataset.get_predict_data_loader()
