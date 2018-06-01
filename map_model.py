'''
Created on May 31, 2018

@author: eran
'''
import torch.nn as nn
from torch.autograd import Variable 
import math
from torch.nn import functional
from torchvision.models import resnet

class ClassNonClassModel(nn.Module):
    '''
    classdocs
    '''

    def __init__(self, num_in_classes=1000):
        '''
        Constructor
        '''
        super(ClassNonClassModel, self).__init__()
        self.num_in_classes = num_in_classes
        self.num_out_classes = 2
        self.fc = nn.Linear(self.num_in_classes, self.num_out_classes)
        
        #initialize model (layer) parameters
        params = list(self.fc.parameters())
        pw = params[0]
        n = (self.num_out_classes * self.num_in_classes)
        pw.data.normal_(0, math.sqrt(2. / n))
        pb = params[1]
        pb.data.zero_()


    def forward(self,x):
        #y = functional.softmax(x).data.squeeze()
        y = functional.softmax(x).data.squeeze()
        y = Variable(y) 
        y.requires_grad = True 
        y = self.fc(x)
        return y 
    
        