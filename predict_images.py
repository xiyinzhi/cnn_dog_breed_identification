# -*- coding:utf-8 -*-
__author__ = 'Yi'

import torch
import torchvision.transforms as transforms
import sys

sys.path.append('/Users/dingyi/workingspace/pytorch_space')


def predict_one_image(imgpath, input_space='RGB'):
    from PIL import Image
    img = Image.open(imgpath).convert(input_space)
    input_data = preprocess(img).unsqueeze(0)
    input_tensor = torch.autograd.Variable(input_data)
    output = model(input_tensor).data.squeeze().cpu()
    score = output.numpy() #shape=(120,)
    # score = torch.sigmoid(output).numpy() #map score to [0,1]
    return score


if __name__ == '__main__':
    num_classes = 120
    trained_model_path = '/Users/dingyi/Desktop/dog_breed/models/pytorch_inception_resnet_v2/best.model'
    model = torch.load(trained_model_path)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose([
        transforms.Scale(311),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize
    ])

    from tqdm import tqdm
    import glob

    data = {}
    imgpaths = glob.glob('/Users/dingyi/Desktop/dog_breed/1/test/*.jpg')
    for x in tqdm(imgpaths):
        pred = predict_one_image(x)
        data[x] = pred

    outf = '/Users/dingyi/Desktop/dog_breed/submit'
    with open('%s/submit.txt' % outf, 'w') as f:
        for x in data.keys():
            f.write(x + '\n')

    scores = []
    for x in data.keys():
        scores.append(data[x])
    import numpy as np

    np.array(scores).dump('%s/scores' % outf)
