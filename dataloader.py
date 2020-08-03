import torch
from torch.utils import data
import pandas as pd
import numpy as np
from smooth_data import preproc
from pathlib import Path

pd.set_option('mode.chained_assignment', None)


def read_df(df_file):
    df = pd.read_hdf(df_file)
    df.columns = df.columns.droplevel(0)
    df.index.name = 'index'
    df.index = df.index.map(int)
    df = df.applymap(float)
    return df


def process_df(df):
    body_parts = pd.unique([col[0] for col in df.columns])
    smoothed_data = {}
    for part in body_parts:
        smoothed_data[(part, 'x')] = preproc(df[part].x, df[part].likelihood)
        smoothed_data[(part, 'y')] = preproc(df[part].y, df[part].likelihood)
        smoothed_data[(part, 'likelihood')] = df[part].likelihood.copy()

    smooth_df = pd.DataFrame.from_dict(smoothed_data)
    return smooth_df


def normalize_coordinates(df: pd.DataFrame):
    N = len(df)
    xy_df = df.drop(axis=1, columns='likelihood', level=1)
    coords = xy_df.values.reshape(N, -1, 2)
    base_tail_coords = xy_df.tailbase.values[:, np.newaxis, :]
    centered_coords = coords - base_tail_coords
    centered_nose_coords = xy_df.nose.values - xy_df.tailbase.values
    thetas = np.arctan2(centered_nose_coords[:, 1], centered_nose_coords[:, 0])
    rotation_matrices = np.stack([np.stack([np.cos(thetas), np.sin(thetas)], axis=-1),
                                  np.stack([np.sin(thetas), -np.cos(thetas)], axis=-1)], axis=-1)
    normalized_coords = np.einsum("nij,nkj->nki", rotation_matrices, centered_coords)
    print(xy_df.head())
    print((centered_coords[1000]))
    print(normalized_coords[1000])
    return normalized_coords


class LandmarkDataset(data.Dataset):
    def __init__(self, landmarks_file):
        super(LandmarkDataset, self).__init__()
        self.file = landmarks_file
        self.df = read_df(landmarks_file)
        self.df = process_df(self.df)
        normalize_coordinates(self.df)


def main():
    data_path = Path('/home/orel/Data/K6/2020-03-30/Down/')
    landmarks_file = data_path / '0014DeepCut_resnet50_DownMay7shuffle1_1030000.h5'
    dataset = LandmarkDataset(landmarks_file)


if __name__ == "__main__":
    main()
