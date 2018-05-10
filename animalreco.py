import os
#import os.path
import sys
import skimage.io
import glob
import torchvision
from torchvision import transforms as trn
from torch.autograd import Variable as V
from torch.nn import functional as f
import time
import logging
import re
from shutil import copyfile


logging.basicConfig(filename='demo.log',level=10)

def check_results(groundTrue, output_path ):
 #   for dirpath, dirname, in os.walk(",");
#
    outOtherFiles = glob.glob(output_path+'/other/'+'*.JPG')
    outTrashyFiles = glob.glob(output_path+'/trashy/'+'*.JPG')
    otherGroundTrueFiles = glob.glob(groundTrue+'/other/'+'*.JPG')
    trashyGroundTrueFiles = glob.glob(groundTrue+'/trashy/'+'*.JPG')
    numPositive = len (otherGroundTrueFiles)
    numNegative = len (trashyGroundTrueFiles)
    falsePositive = 0
    truePositive = 0
    trueNegative = 0
    falseNegative = 0
    #import pdb;pdb.set_trace()
    for file in outOtherFiles:
        file = groundTrue+"/other/"+os.path.basename(file)
        print(file)
        if file in otherGroundTrueFiles:
            truePositive = truePositive+1
        else:
            falsePositive = falsePositive +1
    for file in outTrashyFiles:
        file = groundTrue+"/trashy/"+os.path.basename(file)
        if file in trashyGroundTrueFiles:
            trueNegative = trueNegative+1
        else:
            falseNegative = falseNegative+1
    print("{},{},{},{:.3f}".format("TruePositive/numPositive",truePositive,numPositive,float(truePositive)/float(numPositive+1.0)))
    print("{},{},{},{:.3f}".format("FalseNegative/numPositive",falseNegative,numPositive,float(falseNegative)/float(numPositive+1.0)))
    str1 = "falseNegative:"+str(falseNegative)+",trueNegative:"+str(trueNegative)+",numNegative:"+str(numNegative) 
    str2 = "falsePositive:"+str(falsePositive)+",truePositive:"+str(truePositive)+",numPositive"+str(numPositive) 
    logging.basicConfig(filename='result.log')
    log = logging.getLogger("result")
    log.info(output_path)
    log.info(str1)
    log.info(str2)
    print(str1+"\n"+str2)

def get_model(name,cuda=True):
    name2model = {"resnet18":torchvision.models.resnet18,
                  "resnet34":torchvision.models.resnet34,
                  "resnet50":torchvision.models.resnet50,
                  "resnet101":torchvision.models.resnet101,
                  "resnet152":torchvision.models.resnet152,
                 }
    if name not in name2model:
        raise Exception("missing model {}".format(name))
    log = logging.getLogger("get_model")
    log.info("Working with model={}".format(name))
    model = name2model[name](pretrained=True)
    model.eval()
    if cuda:
        model.cuda()
    return model

def main():
    search_pattern = sys.argv[1]
    dest =  sys.argv[2]
    modelname = "resnet152"
    output_path = '/tmp/test-'+modelname+'/'+dest
    groundTrue = sys.argv[3]#'/tmp/test-resnet152/X18GT/'
    #check_results(groundTrue,output_path )
    #exit()
    log = logging.getLogger("main")

    # get classes
    file_name = 'synset_words.txt'
    classes = get_synset(file_name)

    # define image transformation
    CenterCropList =[]
    centre_crop = trn.Compose([
            trn.ToPILImage(),
            trn.Scale(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #CenterCropList.append(centre_crop)
    centre_crop = trn.Compose([
            trn.ToPILImage(),
            trn.Scale(256),
    #trn.ColorJitter( brightness=0, contrast=0.2, saturation=0, hue=0),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #CenterCropList.append(centre_crop)
    centre_crop = trn.Compose([
            trn.ToPILImage(),
            trn.Scale(256),
            trn.ColorJitter(brightness=0, contrast=0.4, saturation=0, hue=0),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #CenterCropList.append(centre_crop)
    centre_crop = trn.Compose([
            trn.ToPILImage(),
            trn.Scale(256),
            #trn.RandomGrayscale(),
            trn.RandomHorizontalFlip(),
            trn.RandomCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    CenterCropList.append(centre_crop)
    centre_crop = trn.Compose([
            trn.ToPILImage(),
            trn.Scale(256),
            #trn.ColorJitter(brightness=0, contrast=0.8, saturation=0, hue=0),
            trn.CenterCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    CenterCropList.append(centre_crop)





    # get top 5 probabilities
    files = glob.glob(search_pattern) 
    log.info("Looking at {} images".format(len(files))) 
    model = get_model(modelname)
    print(len(files))
    th = 0.3
    found = 0
    for file_name in files:
        #print("Working on {}".format(file_name))
        log.info("Working on {}".format(file_name))
        img = skimage.io.imread(file_name)
        t0 = time.time()
        #x = V(centre_crop(img).unsqueeze(0), volatile=True)

        bestprob =0
        bestidx = 0
        for cc in CenterCropList: 
            x = V(cc(img).unsqueeze(0), volatile=True)

        #import pdb;pdb.set_trace()
            x = x.cuda()
            logit = model.forward(x)
            h_x = f.softmax(logit).data.squeeze()
            probs, idx = h_x.sort(0, True)
            if probs[0] > bestprob:
                bestprob = probs[0]
                bestidx  = idx[0]
        if bestprob  > th :
            c = 'other'
            d= classes[bestidx]
            found = found + 1
        else:
            c = 'trashy'
        print(c)
        dest = os.path.join(output_path, c)
        if not os.path.isdir(dest):
            os.makedirs(dest)
        basename = os.path.basename(file_name)
        basename = re.sub('\(|\)|\ ',"_",str(basename))
        dest = os.path.join(dest,basename)
        copyfile(file_name,dest)
        pbs = ",".join(["{:.3f}".format(a0) for a0 in probs[:5]])
        #d = ",".join(["{}".format(a0) for a0 in classes[idx[:5]]])
        print("{},{:.3f},{}".format(file_name[-10:],probs[0],classes[idx[0]]))
        #for i in range(0, 2):
        #   print('{:.3f} -> {}'.format(probs[i], classes[idx[i]] ))
        #log.info("took {} secs".format(time.time()-t0))
        #print("took {} secs".format(time.time()-t0))
    print("{}, {}".format("Found",found))
    check_results(groundTrue,output_path )

def get_synset(file_name):
    if not os.access(file_name, os.W_OK):
        synset_URL = 'https://github.com/szagoruyko/functional-zoo/raw/master/synset_words.txt'
        os.system('wget ' + synset_URL)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ', 1)[1].split(', ', 1)[0])
    classes = tuple(classes)
    return classes




if __name__ == '__main__':
    main()
