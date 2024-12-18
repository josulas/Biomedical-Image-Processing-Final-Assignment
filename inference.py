import numpy as np
from segmentation.k_means import kmeans, KmeansFlags, KmeansTermCrit, KmeansTermOpt
from improving.filtering import conv2d
import cv2
from numpy.typing import NDArray, ArrayLike
from torch import nn, optim
import torch.nn.functional as functional
import torch
import platform
import numpy as np

MODEL_PATH = 'model_cpu.pth'

PREDICTION_POWER = np.array([0.99404762, 1., 0.98589342, 0.98257081]) * 100

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        first_stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=first_stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        if downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
            self.downsample_bn = nn.BatchNorm2d(out_channels)
    def forward(self, input_vec):
        conv1_out = self.conv1(input_vec)
        bn1_out = self.bn1(conv1_out)
        relu1_out = self.relu1(bn1_out)
        conv2_out = self.conv2(relu1_out)
        bn2_out = self.bn2(conv2_out)
        if self.downsample:
            input_vec = self.downsample_conv(input_vec)
            input_vec = self.downsample_bn(input_vec)
        output = input_vec + bn2_out
        return self.relu2(output)


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    def forward(self, x):
        return self.sequential(x)


def get_classification_layer(in_channels, out_channels=1000):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(in_features=in_channels, out_features=out_channels),
    )

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = Stem()
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 64, downsample=False),
            ResidualBlock(64, 64, downsample=False)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 128, downsample=False)
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(128, 256, downsample=True),
            ResidualBlock(256, 256, downsample=False)
        )
        self.stage5 = nn.Sequential(
            ResidualBlock(256, 512, downsample=True),
            ResidualBlock(512, 512, downsample=False)
        )
        self.classification_layer = get_classification_layer(512)
    def forward(self, input_vec):
        stage1_output = self.stage1(input_vec)
        stage2_output = self.stage2(stage1_output)
        stage3_output = self.stage3(stage2_output)
        stage4_output = self.stage4(stage3_output)
        stage5_output = self.stage5(stage4_output)
        classification = self.classification_layer(stage5_output)
        return classification


class CustomResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.skeleton = nn.Sequential(*list(ResNet18().children())[:-1])
        self.classifier = get_classification_layer(512, 4)
    def forward(self, input_vec):
        return self.classifier(self.skeleton(input_vec))
    def set_requires_grad_skeleton(self, requires_grad: bool = True):
        for param in self.skeleton.parameters():
            param.requires_grad = requires_grad


def setup_device():
    if platform.system() == "Windows":
        # For Windows, use torch_directml if available
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            # Fallback to CPU if torch_directml is not installed
            device = torch.device("cpu")
    elif platform.system() == "Darwin":
        # For macOS, use MPS (Metal Performance Shaders) if available, otherwise CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        # Fallback for other platforms
        device = torch.device("cpu")
    return device


class MainModel(nn.Module):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999):
        super().__init__()
        self.device = setup_device()
        self.net = CustomResNet18().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.net.parameters(), lr=lr, betas=(beta1, beta2))
        self.input = None
        self.target = None
        self.prediction = None
        self.ce_loss = None

    def set_requires_grad(self, requires_grad=True):
        for p in self.net.parameters():
            p.requires_grad = requires_grad
    def setup_input(self, data):
        data = torch.tensor(data).to(self.device)
        self.input = data[0]
        self.target = data[1]
    def setup_input_no_label(self, data):
        self.input = torch.tensor(data).to(self.device)
    def forward(self):
        self.prediction = self.net(self.input)
    def backward(self):
        self.ce_loss = self.criterion(self.prediction, self.target)
        self.ce_loss.backward()
    def optimize(self):
        self.forward()
        self.net.train()
        self.opt.zero_grad()
        self.backward()
        self.opt.step()



def build_and_load_model_from_state(path: str = MODEL_PATH):
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    model = MainModel()
    # serialization.add_safe_globals([
    #     torch._utils._rebuild_device_tensor_from_numpy,
    #     np.core.multiarray._reconstruct,
    #     np.ndarray,
    #     _codecs.encode,
    #     np.dtype
    # ])
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def iter_list(func):
    def wrapper(lst, *args, **kwargs):
        results = []
        for element in lst:
            results.append(func(element, *args, **kwargs))
        return np.array(results)
    return wrapper


@iter_list
def pre_process_image(image_array: ArrayLike) -> NDArray:
    epsilon = 0.1
    sigma = 3
    gaussian_dim = int(np.ceil(np.sqrt(-2 * sigma ** 2 * np.log(epsilon * sigma * np.sqrt(2 * np.pi)))))
    gaussian_kernel1_d = cv2.getGaussianKernel(gaussian_dim, sigma)
    gaussian_kernel = np.outer(gaussian_kernel1_d, gaussian_kernel1_d)
    filtered_image = conv2d(image_array, gaussian_kernel)
    compactness, labels_kmeans, centers = kmeans(
        filtered_image.flatten(), 3,
        criteria=KmeansTermCrit(KmeansTermOpt.BOTH, 20, 0.5),
        flags=KmeansFlags.KMEANS_PP_CENTERS, attempts=5
    )
    centers = centers.astype(np.uint8)
    segmented_kmeans = centers[labels_kmeans].reshape(image_array.shape)
    sorted_centers = sorted(centers)
    white_matter_idx = np.argmax(centers == sorted_centers[2])
    grey_matter_idx = np.argmax(centers == sorted_centers[1])
    segmented_white_matter = torch.tensor(np.where(segmented_kmeans == centers[white_matter_idx], 1, 0),
                                          dtype=torch.float)
    segmented_grey_matter = torch.tensor(np.where(segmented_kmeans == centers[grey_matter_idx], 1, 0),
                                         dtype=torch.float)
    image_array = torch.tensor(image_array.astype(np.float64) / 255, dtype=torch.float)
    return np.array((image_array, segmented_white_matter, segmented_grey_matter))


def predict(model: MainModel, input_vec: ArrayLike) -> NDArray:
    with torch.no_grad():
        model.setup_input_no_label(input_vec)
        model.forward()
    predicted = np.argmax(functional.softmax(model.prediction.detach().cpu(), dim=1), axis=1)
    return np.array(predicted)


# Example Usage
if __name__ == '__main__':
    from dataset import load_dataset, TEST_DATASET, get_elements_from_indexes
    from sklearn.metrics import classification_report
    dataset_dict = load_dataset(**TEST_DATASET)
    length = len(dataset_dict)
    n_test = length
    sample_indexes = np.random.choice(length, n_test, replace=False)
    sample = get_elements_from_indexes(dataset_dict, sample_indexes)
    images = [element[0] for element in sample]
    labels = np.array([element[1] for element in sample])
    classifier = build_and_load_model_from_state()
    inputs = pre_process_image(images)
    prediction = predict(classifier, inputs)
    report = classification_report(labels, prediction)
    print(f"Model performance over {n_test} random images from the validation set:")
    print(report)
    # Predictive Power calculation
    prediction = np.array(prediction)
    correctly_classified = np.array([np.sum((labels == i) & (prediction == i)) for i in range(4)])
    total_per_class = np.array([np.sum(prediction == i) for i in range(4)])
    predictive_power = correctly_classified / total_per_class
    print("Predictive Power:")
    print(predictive_power)
