"""This module contains the inference workflow for image classification."""


import os
import platform
from joblib import Parallel, delayed

import numpy as np
from numpy.typing import NDArray, ArrayLike
import cv2
import torch
from torch import nn
import torch.nn.functional as functional
from torchvision.models import efficientnet_b0
from datasets import load_dataset,  DatasetDict
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from dotenv import load_dotenv

from libraries.segmentation.k_means import kmeans, KmeansFlags, KmeansTermCrit, KmeansTermOpt
from libraries.improving.filtering import conv2d


# Load environment variables from .env file
load_dotenv()


MODEL_SD_FILE_NAME = 'model_cpu.pth'
TEST_DATASET = {'path': 'Falah/Alzheimer_MRI',
                'split': 'test'}
LABELS = {0: "Mild_Demented",
          1: "Moderate_Demented",
          2: "Non_Demented",
          3: "Very_Mild_Demented"}
PREDICTION_POWER_FILE_NAME = \
    os.environ.get('PREDICTION_POWER_NUMPY_FILE', 'prediction_power.npy')


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


def setup_device():
    """
    Set up the device for PyTorch based on the system configuration.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif platform.system() == "Windows":
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            device = torch.device("cpu")
    elif platform.system() == "Darwin":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device


def get_classification_layer_efficientnet(in_channels=1280, out_channels=1000):
    """
    Create a classification layer for EfficientNet models.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output classes.
    Returns:
        nn.Sequential: A sequential model containing the classification layer.
    """
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(in_features=in_channels, out_features=out_channels),
    )


class CustomEfficientNetB0Inference(nn.Module):
    """
    Custom EfficientNetB0 model for image classification.
    This model uses EfficientNetB0 as the backbone and adds a custom classification layer.
    """
    def __init__(self):
        super().__init__()
        self.skeleton =  nn.Sequential(*list(efficientnet_b0(weights=None).children())[:-1])
        self.classifier = get_classification_layer_efficientnet(1280, 4)
    def forward(self, input):
        """Execute the forward pass of the model."""
        return self.classifier(self.skeleton(input))


class MainModelInference(nn.Module):
    """Main model for inference."""
    def __init__(self):
        super().__init__()
        self.device = setup_device()
        self.net = CustomEfficientNetB0Inference().to(self.device)  # Changed this line
        self.input = None
        self.prediction = None
    def set_requires_grad(self, requires_grad=True):
        """Set the requires_grad attribute for all parameters in the model."""
        for p in self.net.parameters():
            p.requires_grad = requires_grad
    def setup_input(self, data):
        """Set up the input tensor for the model."""
        if isinstance(data, list):
            data = np.stack(data)
        elif isinstance(data, np.ndarray):
            pass
        else:
            data = np.array(data)
        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)
        self.input = torch.from_numpy(data).float().to(self.device)
    def forward(self):
        """Execute the forward pass of the model."""
        self.prediction = self.net(self.input)


def build_and_load_model_from_state(path: str = MODEL_SD_FILE_NAME):
    """
    Build and load the model from a saved state dictionary.
    """
    model = MainModelInference()
    model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
    return model


def iter_list(func):
    """Decorator to apply a function to each element in a list."""
    def wrapper(lst, *args, **kwargs):
        if not isinstance(lst, list):
            return func(lst, *args, **kwargs)
        results = []
        for element in lst:
            results.append(func(element, *args, **kwargs))
        return np.stack(results)
    return wrapper


@iter_list
def pre_process_image(image_array: NDArray) -> NDArray:
    """
    Pre-process the input image by applying Gaussian filtering and K-means segmentation.
    Args:
        image_array (ArrayLike): Input image as a NumPy array.
    Returns:
        NDArray: A tuple containing the normalized image, segmented white matter,
        and segmented grey matter.
    """
    epsilon = 0.1
    sigma = 3
    gaussian_dim = int(np.ceil(np.sqrt(-2 * sigma ** 2 * \
                        np.log(epsilon * sigma * np.sqrt(2 * np.pi)))))
    gaussian_kernel1_d = cv2.getGaussianKernel(gaussian_dim, sigma)
    gaussian_kernel = np.outer(gaussian_kernel1_d, gaussian_kernel1_d)
    filtered_image = conv2d(image_array, gaussian_kernel)
    _, labels_kmeans, centers = kmeans(
        filtered_image.flatten(), 3,
        criteria=KmeansTermCrit(KmeansTermOpt.BOTH, 20, 0.5),
        flags=KmeansFlags.KMEANS_PP_CENTERS, attempts=5
    )
    centers = centers.astype(np.uint8)
    segmented_kmeans = centers[labels_kmeans].reshape(image_array.shape)
    sorted_centers = sorted(centers)
    white_matter_idx = np.argmax(centers == sorted_centers[2])
    grey_matter_idx = np.argmax(centers == sorted_centers[1])
    segmented_white_matter = \
        np.where(segmented_kmeans == centers[white_matter_idx], 1, 0).astype(np.float64)
    segmented_grey_matter = \
        np.where(segmented_kmeans == centers[grey_matter_idx], 1, 0).astype(np.float64)
    image_array = image_array.astype(np.float64) / 255
    return np.array((image_array, segmented_white_matter, segmented_grey_matter))


def predict(model: MainModelInference, input_vec: NDArray) -> NDArray:
    """
    Predict the class of the input vector using the model.
    Args:
        model (MainModelInference): The model to use for prediction.
        input_vec (ArrayLike): The input vector to classify.
    Returns:
        NDArray: The predicted class labels.
    """
    with torch.no_grad():
        model.setup_input(input_vec)
        model.forward()
    predicted = np.argmax(functional.softmax(model.prediction.detach().cpu(), dim=1), axis=1)
    return predicted.numpy()


def load_test_dataset() -> tuple[list[NDArray], NDArray]:
    """
    Load the test dataset and return images and labels.
    Returns:
        tuple[list[NDArray], NDArray]: A tuple containing a list of images and an array of labels.
    """
    dataset_dict = load_dataset(**TEST_DATASET)
    sample_indexes = np.arange(len(dataset_dict))
    sample = get_elements_from_indexes(dataset_dict, sample_indexes)
    images = [element[0] for element in sample]
    labels = np.array([element[1] for element in sample])
    return images, labels


def get_predictive_power(image_classifier: MainModelInference,
                         images: list[NDArray],
                         labels: NDArray) -> tuple[NDArray, NDArray]:
    """Get the predictive power of the model on a test dataset."""
    inputs = Parallel(n_jobs=8)(delayed(pre_process_image)(element) for element in tqdm(images))
    prediction = predict(image_classifier, inputs)
    prediction = np.array(prediction)
    correctly_classified = np.array([np.sum((labels == i) & (prediction == i)) for i in range(4)])
    total_per_class = np.array([np.sum(prediction == i) for i in range(4)])
    predictive_power = correctly_classified / total_per_class
    return prediction, predictive_power


if __name__ == '__main__':
    test_images, test_labels = load_test_dataset()
    classifier = build_and_load_model_from_state()
    predicted_labels, power = get_predictive_power(classifier, test_images, test_labels)
    report = classification_report(test_labels, predicted_labels)
    print(f"Model performance over {len(test_labels)} all images from the validation set:")
    print(report)
    print(f"Predictive power per class: {power}")
    np.save(PREDICTION_POWER_FILE_NAME, power)
