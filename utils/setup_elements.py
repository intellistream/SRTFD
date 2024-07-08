import torch
from models.resnet import Reduced_ResNet18, SupConResNet
from models.transformers import TransformerFeatureExtractor
from models.network import GCFAggMVC
from models.GurModel import NetworkModel
from torchvision import transforms
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
default_trick = {'labels_trick': False, 'kd_trick': False, 'separated_softmax': False,
                 'review_trick': False, 'ncm_trick': False, 'kd_trick_star': False}


input_size_match = {
    'cifar100': [3, 32, 32],
    'cifar10': [32, 32, 3],
    'core50': [3, 128, 128],
    'mini_imagenet': [3, 84, 84],
    'openloris': [3, 50, 50],
    'HRS': [120],
    'TEP': [52],
    'CARLS_S': [10],
    'CARLS_M': [10],
}


n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'core50': 50,
    'mini_imagenet': 100,
    'openloris': 69,
    'HRS': 6,
    'TEP': 22,
    'CARLS_S': 10,
    'CARLS_M': 5,
}


transforms_match = {
    'core50': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToTensor()]),
    'openloris': transforms.Compose([
        transforms.ToTensor()]),
    'HRS': transforms.Compose([
        transforms.ToTensor()]),
    'TEP': transforms.Compose([
        transforms.ToTensor()]),
    'CARLS_S': transforms.Compose([
        transforms.ToTensor()]),
    'CARLS_M': transforms.Compose([
        transforms.ToTensor()])
}


def setup_architecture(params):
    nclass = n_classes[params.data]
    if params.agent in ['SCR', 'SCP']:
        if params.data == 'mini_imagenet':
            return SupConResNet(640, head=params.head)
        return SupConResNet(head=params.head)
    if params.agent == 'CNDPM':
        from models.ndpm.ndpm import Ndpm
        return Ndpm(params)
    if params.data == 'cifar100':
        return Reduced_ResNet18(nclass)
    elif params.data == 'cifar10HRS':
        return Reduced_ResNet18(nclass)
    elif params.data == 'core50':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(2560, nclass, bias=True)
        return model
    elif params.data == 'mini_imagenet':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(640, nclass, bias=True)
        return model
    elif params.data == 'openloris':
        return Reduced_ResNet18(nclass)
    elif params.data == 'HRS':
        model = GCFAggMVC(120, 120, nclass, device)
        #model = NetworkModel(120, nclass)
        return model
    elif params.data == 'TEP':
        model = GCFAggMVC(52, 52, nclass, device)
        #model = NetworkModel(52, nclass)
        return model
    elif params.data == 'CARLS_S' or params.data == 'CARLS_M':
        model = GCFAggMVC(10, 10, nclass, device)
        return model


def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                weight_decay=wd)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim
