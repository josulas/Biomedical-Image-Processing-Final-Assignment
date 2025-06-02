"""
Dataset handling for Alzheimer MRI dataset.
This module provides functions to load images and labels from the Alzheimer MRI dataset,
and to retrieve images and labels based on specified indexes.
"""


from datasets import load_dataset, DatasetDict
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


TRAIN_DATASET = {'path': 'Falah/Alzheimer_MRI',
                 'split': 'train'}
TEST_DATASET = {'path': 'Falah/Alzheimer_MRI',
                'split': 'test'}
LABELS = {0: "Mild_Demented",
          1: "Moderate_Demented",
          2: "Non_Demented",
          3: "Very_Mild_Demented"}


def get_images_from_indexes(dataset: DatasetDict, indexes) -> list[NDArray]:
    """
    Get images from the dataset based on the provided indexes.
    Args:    
        dataset (DatasetDict): The dataset containing images.
        indexes (list): List of indexes to retrieve images from.
    Returns:
        list[NDArray]: A list of images as NumPy arrays.
    """
    list_images = []
    pillow_images = dataset[indexes]['image']
    if isinstance(pillow_images, list):
        for image in pillow_images:
            list_images.append(np.array(image))
    else:
        list_images.append(np.array(pillow_images))
    return list_images


def get_labels_from_indexes(dataset: DatasetDict, indexes) -> list[int]:
    """
    Get labels from the dataset based on the provided indexes.
    Args:
        dataset (DatasetDict): The dataset containing labels.
        indexes (list): List of indexes to retrieve labels from.
    Returns:
        list[int]: A list of labels corresponding to the images.
    """
    labels: list[int] | int = dataset[indexes]['label']
    if isinstance(labels, list):
        return labels
    elif isinstance(labels, int):
        # If a single label is returned, wrap it in a list
        return [labels]
    else:
        raise ValueError(f"Labels must be a list or an integer, got: {type(labels)}")


def get_elements_from_indexes(dataset: DatasetDict, indexes) -> list[NDArray]:
    """
    Get elements (images and labels) from the dataset based on the provided indexes.
    Args:
        dataset (DatasetDict): The dataset containing images and labels.
        indexes (list): List of indexes to retrieve elements from.
    Returns:
        list[(NDArray, int)]: A list of tuples, each containing an image
        and its corresponding label.
    """
    list_elements = []
    list_images = get_images_from_indexes(dataset, indexes)
    list_labels = get_labels_from_indexes(dataset, indexes)
    for image, label in zip(list_images, list_labels):
        list_elements.append((image, label))
    return list_elements


if __name__ == '__main__':
    N_SHOW = 4
    dataset_dict = load_dataset(**TEST_DATASET)
    length = len(dataset_dict)    
    sample_indexes = np.random.choice(length, N_SHOW, replace=False)
    sample = get_elements_from_indexes(dataset_dict, sample_indexes)
    fig, axs = plt.subplots(1, N_SHOW, figsize=(20, 5.3), num='Sample')
    for i in range(N_SHOW):
        axs[i].imshow(sample[i][0], cmap='gray', vmin=0, vmax=255)
        axs[i].set_title(LABELS[sample[i][1]])
        axs[i].axis('off')
    fig.suptitle('Sample Images')
    fig.tight_layout()
    fig.show()
    input()
