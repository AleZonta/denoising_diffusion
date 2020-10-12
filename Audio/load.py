"""
TLSTM. Turing Learning system to generate trajectories
Copyright (C) 2018  Alessandro Zonta (a.zonta@vu.nl)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import torch
from sklearn.preprocessing import minmax_scale
from torchvision import transforms
import torch.utils.data as data_utils

from Audio.CustomTensor import CustomTensorDataset
from Audio.TorchLoader import TorchLoader


class PadNumpy(object):

    def __init__(self, type_padding):
        self._type_padding = type_padding

    def __call__(self, sample):
        real_shape = sample.shape
        if real_shape[2] == 49:
            sample = np.pad(sample, ((0, 0), (0, 0), (8, 7)), self._type_padding)
        if real_shape[2] == 40:
            sample = np.pad(sample, ((0, 0), (0, 0), (12, 12)), self._type_padding)
        if real_shape[1] == 25:
            sample = np.pad(sample, ((0, 0), (20, 19), (0, 0)), self._type_padding)
        if real_shape[2] == 99:
            sample = np.pad(sample, ((0, 0), (0, 0), (15, 14)), self._type_padding)
        if real_shape[2] == 80:
            sample = np.pad(sample, ((0, 0), (0, 0), (24, 24)), self._type_padding)
        if real_shape[1] == 50:
            sample = np.pad(sample, ((0, 0), (39, 39), (0, 0)), self._type_padding)
        return sample

def load(args):
    """Load dataset.
    Args:
        dataset: name of dataset.
    Returns:
        a torch dataset and its associated information.
    """
    audio = TorchLoader(path_source=args.source_folder, sample_rate=args.sample_rate)
    x_train, train_y, x_val, val_y, x_test, test_y = audio.load_data(
        source_path=args.source_folder).load_files().save_data(
        destination_path=args.source_folder).get_data()

    x_train = minmax_scale(x_train)
    x_val = minmax_scale(x_val)
    x_test = minmax_scale(x_test)

    x_train = np.reshape(x_train, (-1, args.in_channels, args.x, args.y))
    x_val = np.reshape(x_val, (-1, args.in_channels, args.x, args.y))
    x_test = np.reshape(x_test, (-1, args.in_channels, args.x, args.y))

    x = args.x
    y = args.y
    if y == 49:
        size = 64
    else:
        size = 128
    if y == 40:
        size = 64


    data_transform_pad = transforms.Compose([
        PadNumpy("constant"),
    ])

    train_dataset = CustomTensorDataset(torch.from_numpy(x_train), torch.from_numpy(x_train),
                                        transform=data_transform_pad)

    validation_dataset = CustomTensorDataset(torch.from_numpy(x_val), torch.from_numpy(x_val),
                                             transform=data_transform_pad)

    test_dataset = CustomTensorDataset(torch.from_numpy(x_test), torch.from_numpy(x_test), transform=data_transform_pad)

    return train_dataset, validation_dataset, test_dataset
