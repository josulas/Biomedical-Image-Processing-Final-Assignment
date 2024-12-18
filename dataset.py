from datasets import load_dataset, DatasetDict
import numpy as np
from numpy.typing import NDArray


TRAIN_DATASET = {'path': 'Falah/Alzheimer_MRI',
                 'split': 'train'}
TEST_DATASET = {'path': 'Falah/Alzheimer_MRI',
                'split': 'test'}
LABELS = {0: "Mild_Demented",
          1: "Moderate_Demented",
          2: "Non_Demented",
          3: "Very_Mild_Demented"}


def get_images_from_indexes(dataset: DatasetDict, indexes) -> list[NDArray]:
    list_images = []
    pillow_images = dataset[indexes]['image']
    if type(pillow_images) == list:
        for image in pillow_images:
            list_images.append(np.array(image))
    else:
        list_images.append(np.array(pillow_images))
    return list_images


def get_labels_from_indexes(dataset: DatasetDict, indexes) -> list[int]:
    labels = dataset[indexes]['label']
    return labels if type(labels) == list else [labels]


def get_elements_from_indexes(dataset: DatasetDict, indexes) -> list[(NDArray, int)]:
    list_elements = []
    list_images = get_images_from_indexes(dataset, indexes)
    list_labels = get_labels_from_indexes(dataset, indexes)
    for image, label in zip(list_images, list_labels):
        list_elements.append((image, label))
    return list_elements


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    import os
    import cv2
    # Print the number of examples and the first few samples
    dataset_dict = load_dataset(**TEST_DATASET)
    length = len(dataset_dict)
    # Show
    n_show = 4
    sample_indexes = np.random.choice(length, n_show, replace=False)
    sample = get_elements_from_indexes(dataset_dict, sample_indexes)
    fig, axs = plt.subplots(1, n_show, figsize=(20, 5.3), num='Sample')
    for i in range(n_show):
        axs[i].imshow(sample[i][0], cmap='gray', vmin=0, vmax=255)
        axs[i].set_title(LABELS[sample[i][1]])
        axs[i].axis('off')
    fig.suptitle('Sample Images')
    fig.tight_layout()
    fig.show()
    # Save two of each type
    save_path = r"interface/images_database"
    entire_test_collection = get_elements_from_indexes(dataset_dict, np.arange(length))
    counts = {}
    n_per_class = 3
    for label in LABELS:
        counts[label] = 0
    for index in range(length):
        if counts[entire_test_collection[index][1]] < n_per_class:
            counts[entire_test_collection[index][1]] += 1
            filename = os.path.join(save_path, f"{LABELS[entire_test_collection[index][1]]}_{counts[entire_test_collection[index][1]]}.png")
            Image.fromarray(entire_test_collection[index][0], "L").save(filename)
    input()

