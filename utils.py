
import cv2
import numpy as np
import pandas as pd

def img_specs0(path):
    img = cv2.imread(path)
    return img.shape[0]
def img_specs1(path):
    img = cv2.imread(path)
    return img.shape[1]
def img_specs2(path):
    img = cv2.imread(path)
    return img.shape[2]

def createDF():
    path1 = "dataset/zebra/zebra_annotation.csv"
    path2 = "dataset/bay horse/bay horse_annotation.csv"
    dfs = []
    tmp = pd.read_csv(path1, sep= ',', header=None)
    dfs.append(tmp)
    tmp = pd.read_csv(path2, sep= ',', header=None)
    dfs.append(tmp)
    df = pd.concat(dfs)
    df.drop(1, axis=1, inplace=True)
    df.rename(columns={0:'path', 2:'class'}, inplace=True)

    df['height'] = df['path'].apply(img_specs0)
    df['width'] = df['path'].apply(img_specs1)
    df['channels'] = df['path'].apply(img_specs2)

