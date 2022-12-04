
from typing import Type
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def createDF() -> pd.DataFrame:
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

    df['mark'] = (df['img_class'] != 'zebra') * int(1)
    df['height'] = df['path'].apply(img_height)
    df['width'] = df['path'].apply(img_width)
    df['channels'] = df['path'].apply(img_channels)

    return df


def df_filter_class(dfSrc: pd.DataFrame, class_name: str) -> pd.DataFrame:
    '''filter using class name as referens'''
    return dfSrc[dfSrc.img_class == class_name]


def df_filter_dimentions(dfSrc: pd.DataFrame, class_name: str, h: int, w: int) -> pd.DataFrame:
    '''filter df to df with certain class name and h and w'''
    return dfSrc[dfSrc.height <= h and dfSrc.width <= w and dfSrc.img_class == class_name]


def stats_with_pixels(df: pd.DataFrame, class_name: str) -> None:
    df = df_filter_class(df, class_name)
    df['pixel'] = df['path'].apply(img_pixels)
    df.groupby('pixel').count()
    df.pixel.describe()


def create_histogram(df: pd.DataFrame, class_name: str) -> Type:
    '''create array contained b g r channels'''
    df = df_filter_class(df, class_name)
    # print(df.sample().iloc[0]['path'])
    image = cv2.imread(df.sample().iloc[0]['path'])
    color = ('b', 'g', 'r')
    result = [[], []]
    for i, col in enumerate(color):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        result[0].append(histr)
        result[1].append(col)
    return result


def draw_histrogram(df: pd.DataFrame, class_name: str) -> None:
    '''draw histogram'''
    tmp = create_histogram(df, class_name)
    for i in range(len(tmp[0])):
        plt.plot(tmp[0][i], color=tmp[1][i])
        plt.xlim([0, 256])
    plt.xlabel("Intensity")
    plt.ylabel("Number of pixels")
    plt.show()


if __name__ == '__main__':
    test = createDF()
    stats_with_pixels(test, 'zebra')