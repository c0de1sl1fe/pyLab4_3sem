
from typing import Type
import cv2
import numpy as np
import pandas as pd


def img_height(path: str) -> int:
    '''return characteristic of img '''
    img = cv2.imread(path)
    return img.shape[0]


def img_width(path: str) -> int:
    '''return characteristic of img '''
    img = cv2.imread(path)
    return img.shape[1]


def img_channels(path: str) -> int:
    '''return characteristic of img '''
    img = cv2.imread(path)
    return img.shape[2]


def img_pixels(path: str) -> int:
    '''return characteristic of img '''
    img = cv2.imread(path)
    return img.size


def createDF() -> Type(pd):
    '''1 to 5 chapter of lab'''
    path1 = "dataset/zebra/zebra_annotation.csv"
    path2 = "dataset/bay horse/bay horse_annotation.csv"
    dfs = []
    tmp = pd.read_csv(path1, sep=',', header=None)
    dfs.append(tmp)
    tmp = pd.read_csv(path2, sep=',', header=None)
    dfs.append(tmp)
    df = pd.concat(dfs)
    df.drop(1, axis=1, inplace=True)
    df.rename(columns={0: 'path', 2: 'img_class'}, inplace=True)

    df['mark'] = (df['img_class'] != 'zebra')
    df['height'] = df['path'].apply(img_height)
    df['width'] = df['path'].apply(img_width)
    df['channels'] = df['path'].apply(img_channels)

    return df


def df_filter_class(dfSrc: Type(pd), class_name: str) -> Type(pd):
    '''filter using class name as referens'''
    return dfSrc[dfSrc.img_class == class_name]


def df_filter_dimentions(dfSrc: Type(pd), class_name: str, h: int, w: int) -> Type(pd):
    '''filter df to df with certain class name and h and w'''
    return dfSrc[dfSrc.height <= h and dfSrc.width <= w and dfSrc.img_class == class_name]


def second_part(df:Type(pd)) -> None:
    df['pixel'] = df['path'].apply(img_pixels)