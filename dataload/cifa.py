import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

import easydict 

def get_cifa10(args):
    train_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=(-30, 30)),
                                    transforms.RandomErasing(p=0.2),
                                    transforms.RandomAffine(30, shear=20)
                                    ])


    vaild_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

    #데이터 불러오기, 학습여부  o
    trainset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True,
                                            download=True, 
                                            transform=train_transform
                                            )

    #데이터 불러오기, 학습여부  x
    testset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=False,
                                        download=True, 
                                        transform=vaild_transform
                                        )

    dataset_size = len(trainset)
    train_size = int(dataset_size * 0.9)
    validation_size = dataset_size - train_size
    train_dataset, validation_dataset = random_split(trainset, [train_size, validation_size])
    #학습용 셋은 섞어서 뽑기
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=args.BATCH_SIZE,
                                            shuffle=True, 
                                            num_workers=4
                                            )
    
    vaildloader = torch.utils.data.DataLoader(validation_dataset, 
                                            batch_size=args.BATCH_SIZE,
                                            shuffle=True, 
                                            num_workers=4
                                            )

    #테스트 셋은 굳이 섞을 필요가 없음
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=args.TEST_BATCH_SIZE,
                                            shuffle=False, 
                                            num_workers=2
                                            )
    #클래스들
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, vaildloader, testloader, classes

if __name__=="__main__":
    
    args = easydict.EasyDict({
       'num_epochs':50,
       'num_epochs_ae':50,
       'lr':1e-3,
       'lr_ae':1e-3,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'BATCH_SIZE':1024,
       'TEST_BATCH_SIZE':4,
       'pretrain':False,
       'latent_dim':32,
       'normal_class':0
                })
    dataloader_train, dataloader_vaild, dataloader_test, classes = get_cifa10(args)
    
    #이미지 확인하기

    def imshow(img):
        img = img / 2 + 0.5     # 정규화 해제
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # 학습용 이미지 뽑기
    dataiter = iter(dataloader_test)
    images, labels = next(dataiter)

    # 이미지 보여주기
    imshow(torchvision.utils.make_grid(images))

    # 이미지별 라벨 (클래스) 보여주기
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))