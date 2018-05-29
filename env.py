import argparse
import torch
import torch.nn as nn
import shutil
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

"""
Example PyTorch script for finetuning a ResNet model on your own data.

For this example we will use a tiny dataset of images from the COCO dataset.
We have chosen eight types of animals (bear, bird, cat, dog, giraffe, horse,
sheep, and zebra); for each of these categories we have selected 100 training
images and 25 validation images from the COCO dataset. You can download and
unpack the data (176 MB) by running:

wget cs231n.stanford.edu/coco-animals.zip
unzip coco-animals.zip
rm coco-animals.zip

The training data is stored on disk; each category has its own folder on disk
and the images for that category are stored as .jpg files in the category folder.
In other words, the directory structure looks something like this:

coco-animals/
  train/
    bear/
      COCO_train2014_000000005785.jpg
      COCO_train2014_000000015870.jpg
      [...]
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
  val/
    bear/
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
"""

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='coco-animals/train')
    parser.add_argument('--val_dir', default='coco-animals/val')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_epochs1', default=10, type=int)
    parser.add_argument('--num_epochs2', default=10, type=int)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--resume', default='',type=str,metavar='PATH',help='path of last checkpoint')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    return parser



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Environment(object):
    def __init__(self,args):
        self.dtype = torch.FloatTensor
        if args.use_gpu:
            self.dtype = torch.cuda.FloatTensor
        self.init_dbs(args)
        self.init_model(args)
        self.init_optimizer(args)

    def init_dbs(self,args):
          # Use the torchvision.transforms package to set up a transformation to use
          # for our images at training time. The train-time transform will incorporate
          # data augmentation and preprocessing. At training time we will perform the
          # following preprocessing on our images:
          # (1) Resize the image so its smaller side is 256 pixels long
          # (2) Take a random 224 x 224 crop to the scaled image
          # (3) Horizontally flip the image with probability 1/2
          # (4) Convert the image from a PIL Image to a Torch Tensor
          # (5) Normalize the image using the mean and variance of each color channel
          #     computed on the ImageNet dataset.
          #itrain_transform = T.Compose([
          #  T.Scale(256),
          #  T.RandomSizedCrop(224),
          #  T.RandomHorizontalFlip(),
          #  T.ToTensor(),            
          #  T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
          #])
          train_transform = T.Compose([
              #                 T.ToPILImage(),
              T.Scale(256),
              T.CenterCrop(224),
              T.ToTensor(),
              T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
          ])
          # You load data in PyTorch by first constructing a Dataset object which
          # knows how to load individual data points (images and labels) and apply a
          # transform. The Dataset object is then wrapped in a DataLoader, which iterates
          # over the Dataset to construct minibatches. The num_workers flag to the
          # DataLoader constructor is the number of background threads to use for loading
          # data; this allows dataloading to happen off the main thread. You can see the
          # definition for the base Dataset class here:
          # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataset.py
          #
          # and you can see the definition for the DataLoader class here:
          # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py#L262
          #
          # The torchvision package provides an ImageFolder Dataset class which knows
          # how to read images off disk, where the image from each category are stored
          # in a subdirectory.
          #
          # You can read more about the ImageFolder class here:
          # https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
          self.train_dset = ImageFolder(args.train_dir, transform=train_transform)
          self.train_loader = DataLoader(self.train_dset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=True)

          # Set up a transform to use for validation data at test-time. For validation
          # images we will simply resize so the smaller edge has 224 pixels, then take
          # a 224 x 224 center crop. We will then construct an ImageFolder Dataset object
          # for the validation data, and a DataLoader for the validation set.
          val_transform = T.Compose([
              T.Scale(224),
              T.CenterCrop(224),
              T.ToTensor(),
              T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
          ])
          val_dset = ImageFolder(args.val_dir, transform=val_transform)
          self.val_loader = DataLoader(val_dset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)


    def init_model(self,args):
      # First load the pretrained ResNet-18 model; this will download the model
      # weights from the web the first time you run it.
      self.model = torchvision.models.resnet18(pretrained=True)

      # Reinitialize the last layer of the model. Each pretrained model has a
      # slightly different structure, but from the ResNet class definition
      # we see that the final fully-connected layer is stored in model.fc:
      # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L111
      self.num_classes = len(self.train_dset.classes)
      self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

      # Cast the model to the correct datatype, and create a loss function for
      # training the model.
      self.model.type(self.dtype)
      self.loss_fn = nn.CrossEntropyLoss().type(self.dtype)

    def init_optimizer(self,args):
        # First we want to train only the reinitialized last layer for a few epochs.
        # During this phase we do not need to compute gradients with respect to the
        # other weights of the model, so we set the requires_grad flag to False for
        # all model parameters, then set requires_grad=True for the parameters in the
        # last layer only.
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        # Construct an Optimizer object for updating the last layer only.
        #optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
        self.optimizer = torch.optim.SGD(self.model.fc.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    def resume(self,args):
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                if "optimizer" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    def save_checkpoint(self,filename):
        state = {
            'epoch': epoch + 1,
            'arch': 'resnet18',
            'state_dict': self.model.state_dict(),
            'best_prec1': self.best_prec1,
            'optimizer' : self.optimizer.state_dict(),
        }
        torch.save(state, filename)



def train_epoch(model, loss_fn, loader, optimizer, dtype):
  """
  Train the model for one epoch.
  """
  # Set the model to training mode
  model.train()
  bnum = 0
  for x, y in loader:
    # The DataLoader produces Torch Tensors, so we need to cast them to the
    # correct datatype and wrap them in Variables.
    #
    # Note that the labels should be a torch.LongTensor on CPU and a
    # torch.cuda.LongTensor on GPU; to accomplish this we first cast to dtype
    # (either torch.FloatTensor or torch.cuda.FloatTensor) and then cast to
    # long; this ensures that y has the correct type in both cases.
    x_var = Variable(x.type(dtype))
    y_var = Variable(y.type(dtype).long())

    # Run the model forward to compute scores and loss.
    scores = model(x_var)
    loss = loss_fn(scores, y_var)

    # Run the model backward and take a step using the optimizer.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    bnum +=1
    print("finished batch {}".format(bnum))

def check_accuracy(model, loader, dtype):
  """
  Check the accuracy of the model.
  """
  # Set the model to eval mode
  model.eval()
  num_correct, num_samples = 0, 0
  for x, y in loader:
    # Cast the image data to the correct type and wrap it in a Variable. At
    # test-time when we do not need to compute gradients, marking the Variable
    # as volatile can reduce memory usage and slightly improve speed.
    x_var = Variable(x.type(dtype), volatile=True)

    # Run the model forward, and compare the argmax score with the ground-truth
    # category.
    scores = model(x_var)
    _, preds = scores.data.cpu().max(1)
    #print(preds)
    #import pdb
    #import pdb;pdb.set_trace()
    num_correct += (preds == y).sum()
    num_samples += x.size(0)

  # Return the fraction of datapoints that were correctly classified.
  acc = float(num_correct) / num_samples
  print("num_correct:" +str(num_correct)+",num_samples:"+str(num_samples))
  return acc



if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
