import os

import torch
from torchvision import transforms

from loaders.data_list import Imagelists_VISDA, return_classlist


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

def return_dataset(args):
    base_path = './data/txt'
    image_set_file_s = os.path.join(base_path, args.source +'_all' + '.txt')
    image_set_file_t = os.path.join(base_path, args.target + '_labeled' + '.txt')
    image_set_file_test = os.path.join(base_path, args.target + '_unl' + '.txt')
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, transform=data_transforms['train'])
    target_dataset = Imagelists_VISDA(image_set_file_t, transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, transform=data_transforms['val'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset"%len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs, num_workers=3, shuffle=True,
                                                drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=min(bs, len(target_dataset)),
                                                num_workers=3, shuffle=True, drop_last=True)
    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs * 2, num_workers=3,
                                                    shuffle=True, drop_last=True)
    return source_loader, target_loader, target_loader_unl,class_list
def return_dataset_test(args):
    base_path = './data/txt'
    image_set_file_s = os.path.join(base_path, args.source +'_all' + '.txt')
    image_set_file_test = os.path.join(base_path, args.target + '_unl' + '.txt')
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, transform=data_transforms['test'],test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset"%len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs * 2, num_workers=3,
                                                    shuffle=False, drop_last=False)
    return target_loader_unl,class_list
