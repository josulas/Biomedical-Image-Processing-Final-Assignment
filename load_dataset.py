from datasets import load_dataset, DatasetDict
import numpy as np
from numpy.typing import NDArray


TRAIN_DATASET = load_dataset('Falah/Alzheimer_MRI', split='train')
TEST_DATASET = load_dataset('Falah/Alzheimer_MRI', split='test')
LABELS = {0: "Mild_Demented",
          1: "Moderate_Demented",
          2: "Non_Demented",
          3: "Very_Mild_Demented"}


def get_images_from_indexes(dataset: DatasetDict, indexes) -> list[NDArray]:
    list_images = []
    pillow_images = dataset[indexes]['image']
    for image in pillow_images:
        list_images.append(np.array(image))
    return list_images


def get_labels_from_indexes(dataset: DatasetDict, indexes) -> list[int]:
    return dataset[indexes]['label']


def get_elements_from_indexes(dataset: DatasetDict, indexes) -> list[(NDArray, int)]:
    list_elements = []
    list_images = get_images_from_indexes(dataset, indexes)
    list_labels = get_labels_from_indexes(dataset, indexes)
    for image, label in zip(list_images, list_labels):
        list_elements.append((image, label))
    return list_elements


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Print the number of examples and the first few samples
    length = len(TRAIN_DATASET)
    n_show = 4
    sample_indexes = np.random.choice(length, n_show, replace=False)
    sample = get_elements_from_indexes(TRAIN_DATASET, sample_indexes)
    fig, axs = plt.subplots(1, n_show, figsize=(20, 5.3), num='Sample')
    for i in range(n_show):
        axs[i].imshow(sample[i][0], cmap='gray', vmin=0, vmax=255)
        axs[i].set_title(LABELS[sample[i][1]])
        axs[i].axis('off')
    fig.suptitle('Sample Images')
    fig.tight_layout()
    fig.show()

