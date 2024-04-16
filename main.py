import os

import torch
from PIL import Image
from torchvision import models, transforms
from torch import optim
import myData
from simpleCNN import training_model, confusion_matrix_me, EarlyStopping

from imagehash import average_hash


#TODO figure out why this is here below
def get_image_info(self, image_path):
    """
    :param self:
    :param image_path: path of the image
    :return: return the image hash
    """
    # Get information based on the hash of the image
    hash_value = str(average_hash(Image.open(image_path)))
    return self.image_hashes.get(hash_value, None)


def get_model(model_type='resnet18'):
    """
    :param model_type:     this method loads the model depending on the name of that model
    :return: return the pre-trained model
    """
    available_models = {
        '18': models.resnet18(pretrained=True),
        '34': models.resnet34(pretrained=True),
        '50': models.resnet50(pretrained=True),
        '101': models.resnet101(pretrained=True),
        '152': models.resnet152(pretrained=True),
        '121': models.densenet121(pretrained=True),
        '161': models.densenet161(pretrained=True),
        '169': models.densenet169(pretrained=True),
        '201': models.densenet201(pretrained=True),
        'google': models.googlenet(pretrained=True), #only one
        '11': models.vgg11(pretrained=True),
        '19':models.vgg19(pretrained=True),
        '13': models.vgg13(pretrained=True),
        '16':models.vgg16(pretrained=True),
        'alex':models.alexnet(pretrained=True)
    }
    if model_type in available_models:
        return available_models[model_type]
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def get_optim(params, opt='sgd', learn =0.0001, mom =0.9, wd = 1e-4, b1=0.9, b2=0.999, epsil=1e-7):
    """
    :param params: iterable of parameters to optimize or dicts defining parameter groups
    :param opt: the optimizer to use
    :param learn: learning rate
    :param mom: momentum
    :param wd: weight decay
    :param b1: beta 1
    :param b2: beta 2
    :param epsil:epsilon
    :return: return the optimizer if there
    """
    opt = opt.lower()
    available_optims = {
        'sgd': optim.SGD(params, lr=learn, momentum=mom, nesterov=True),
        'asgd': optim.ASGD(params, weight_decay=wd),
        'adagrad': optim.Adagrad(params, weight_decay=wd),
        'adamw': optim.AdamW(params, lr=learn),
        'adam': optim.Adam(params, lr=learn, betas=(b1, b2), eps=epsil),
        'rmsprop':optim.RMSprop(params, lr=learn),
    }
    if opt in available_optims:
        return available_optims[opt]
    else:
        raise ValueError(f"Invalid model type: {opt}")

def rate_images(model, arch, type='hot'):
    """
    rate the image and check that the file type is correct
    :param model: model name
    :param arch: model architecture
    :param type: testing either the hot or not folder
    """
    data = myData.get_architecture(arch)
    class_labels = data.class_labels
    direc = f"NN_data/hot_or_not_oct_23/{type}/"
    cnt_hot, cnt_not = 0, 0
    tot = 0
    avg_hot, avg_not = 0,0
    len_test = len(os.listdir(direc))*0.1
    for idx, file in enumerate(reversed(os.listdir(direc))):
        if idx > len_test: break
        filename = os.fsdecode(file)
        try:
            if file.endswith('.jpg'):
                img = Image.open(direc + filename)
                img = img.convert('RGB')
            else:
                os.remove(direc + filename)
                continue
        except Exception as e:
            os.remove(direc + filename)
            continue
        preprocessed_image = data.transform(img).unsqueeze(0).to('cuda')
        with torch.no_grad():
            logits = model(preprocessed_image)
            _, predicted_class_index = torch.max(logits, 1)
            predicted_label = class_labels[predicted_class_index.item()]
            outputs = torch.nn.functional.softmax(logits, dim=1)

        val = outputs.cpu().numpy()
        if val[0,1] > 0.5:
            cnt_not += 1
        else:
            cnt_hot += 1
        avg_hot += val[0, 0]
        avg_not += val[0, 1]
        # print(f"filename: {filename}")
        # print(f"class_index: {predicted_class_index.item()}")
        # print(f"Predicted Label: {predicted_label}")
        # print(f"Predicted Probabilities: {outputs}")
        # print()
        tot += 1
    print(f"checked folder, {type}")
    print(f"tot: {tot}\nnot: {cnt_not} avg: {avg_not/tot}\nhot: {cnt_hot} avg: {avg_hot/tot}\n")
    
    

def train(name, batch, pocs, model):
    """
    this method trains the model weights
    :param name: name of .pth to save to
    :param batch: batch size
    :param pocs: number of epochs to perform
    :param model: model number to work with
    :return:
    """
    print(f"name: {name}")
    model = get_model(model)
    trainer = training_model()
    trainer.train(name, 'NN_data/hot_or_not_oct_23', batch, pocs, model, optim)
    return name

def test(name, model, arch, type='hot'):
    """
    this method tests the model just trained
    :param name: model .pth file
    :param model: the model from the type of the name ex: resnet18 or resnet34
    :param arch: model architecture ex: resnet vs densenet
    :param type: either test the hot or not folder
    :return:
    """
    model = load_model(name, model, arch)
    rate_images(model, arch, type)
    return model

def load_model(name, type, arch):
    """
    this method loads the model from the type of the name ex: resnet18 or resnet34, its architecture ex: resnet vs densenet
    :param name:
    :param type:
    :param arch:
    :return:
    """
    model = get_model(type)
    if arch == 'res':
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if arch == 'dense':
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    if arch == 'vgg':
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, 2)
    if arch == 'google':
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
    if arch == 'alex':
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(num_ftrs, 2)

    model.load_state_dict(torch.load(name, map_location=torch.device('cuda')))
    model.to('cuda')
    model.eval()
    return model

def removeExtras():
    """
    remove image copies in the training data folder
    :return:
    """
    hashes = set()
    rootdir = 'NN_data/hot_or_not_oct_23'
    cnt = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            image_path = os.path.join(subdir, os.fsdecode(file))
            hash_value = str(average_hash(Image.open(image_path)))
            if hash_value in hashes:
                cnt += 1
                # os.remove(image_path)
            else:
                hashes.add(hash_value)
    print(cnt)


if __name__ == "__main__":
    cm = confusion_matrix_me()

    big = 'NN_data/hot_or_not_oct_23/'
    small = 'nn_smaller/'

    #model/model_type/batch_size/epochs/optimizer/learning_rate
    #d1 = "res_152_64_2_adam_00001_.pth"
    d1 = 'dense_201_64_130_adam_.pth'
    d3 = 'google_google_64_70_adam_.pth'
    # d4 = 'alex_alex_256_100_adam_.pth'


    vals = d1.split('_')
    rm = training_model(big, get_model(vals[1]), vals[0])
    o1 = get_optim(list(rm.model.parameters()), vals[4], learn=0.0001)
    try:
        n1 = rm.train(d1, o1, int(vals[2]), int(vals[3]))
        m1 = test(n1, vals[1], arch=vals[0])
        cm.run(n1, m1, 'NN_data/hot_or_not_oct_23/', 32, vals[0])
    except Exception as e:
        print(e)
        torch.save(rm.model.state_dict(), d1)

    # vals = d2.split('_')
    # rm = training_model(big, get_model(vals[1]), vals[0])
    # o1 = get_optim(list(rm.model.parameters()), vals[4], learn=0.0001)
    # try:
    #     n2 = rm.train(d2, o1, int(vals[2]), int(vals[3]))
    #     m2 = test(n2, vals[1], arch=vals[0])
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32, vals[0])
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d2)
    #
    # vals = d3.split('_')
    # rm = training_model(big, get_model(vals[1]), vals[0])
    # o1 = get_optim(list(rm.model.parameters()), vals[4], learn=0.0001)
    # try:
    #     n2 = rm.train(d3, o1, int(vals[2]), int(vals[3]))
    #     m2 = test(n2, vals[1], arch=vals[0])
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32, vals[0])
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d3)

    # vals = d4.split('_')
    # rm = training_model(small, get_model(vals[1]), vals[0])
    # o1 = get_optim(list(rm.model.parameters()), vals[4], learn=0.0001)
    # try:
    #     n2 = rm.train(d4, o1, int(vals[2]), int(vals[3]))
    #     m2 = test(n2, vals[1], arch=vals[0])
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32, vals[0])
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d4)








