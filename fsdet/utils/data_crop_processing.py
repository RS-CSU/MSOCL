# -*- coding: utf-8 -*-
"""
======================
@author:QinDengda
@time:7/27/21:2:53 PM
@email:qindengda@csu.cn
======================
"""
from torchvision import transforms
from PIL import Image
from PIL import ImageEnhance
import random
import numpy as np
import torch


def data_crop_processing(data):
    data_crop = []
    for img in data:
        box = img['instances']._fields['gt_boxes'].tensor.numpy().tolist()[0]

        img_crop_plt = transforms.ToPILImage()(img['image'][:, int(box[1]):int(box[3]), int(box[0]):int(box[2])])

        img_crop = img_crop_plt.resize((224, 224), Image.BILINEAR).rotate(random.randrange(-45, 45))

        img_array = np.array(img_crop)
        img_tensor = torch.from_numpy(img_array.transpose((2, 0, 1)))

        data_crop.append(img_tensor.float())

    return data_crop
