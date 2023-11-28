import os

import torch
from PIL import Image
from torchvision import models, transforms
from torch import optim
import myData
from simpleCNN import training_model, confusion_matrix_me, EarlyStopping

def get_model(model_type='resnet18'):
    available_models = {
        '18': models.resnet18(pretrained=True),
        '34': models.resnet34(pretrained=True),
        '50': models.resnet50(pretrained=True),
        '101': models.resnet101(pretrained=True),
        '152': models.resnet152(pretrained=True),
        '121': models.densenet121(pretrained=True),
        '161': models.densenet161(pretrained=True),
        '169': models.densenet169(pretrained=True),
        '201': models.densenet201(pretrained=True)
    }
    if model_type in available_models:
        return available_models[model_type]
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def get_optim(params, opt='sgd', learn =0.0001, mom =0.9, wd = 1e-4, b1=0.9, b2=0.999, epsil=1e-7):
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

def bing(model, arch, type='hot'):
    data = myData.get_architecture(arch)
    class_labels = data.class_labels
    direc = f"NN_data/hot_or_not_oct_23/{type}/"
    cnt_hot, cnt_not = 0, 0
    tot = 0
    avg_hot, avg_not = 0,0
    len_test = len(os.listdir(direc))*0.1
    for idx, file in enumerate(os.listdir(direc)):
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
    print(f"name: {name}")
    model = get_model(model)
    trainer = training_model()
    trainer.train(name, 'NN_data/hot_or_not_oct_23', batch, pocs, model, optim)
    return name

def test(name, model, arch, type='hot'):
    model = load_model(name, model, arch)
    bing(model, arch, type)
    return model

def load_model(name, type, arch):
    model = get_model(type)
    if arch == 'res':
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if arch == 'dense':
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(name, map_location=torch.device('cuda')))
    model.to('cuda')
    model.eval()
    return model

if __name__ == "__main__":
    cm = confusion_matrix_me()

    big = 'NN_data/hot_or_not_oct_23/'
    small = 'nn_smaller/'

    d1 = "res_152_32_1_adam_00001_.pth"
    #d2 = "res_152_64_5_adam_0001_.pth"
    d2 = 'dense_121_32_3_adam_.pth'
    d3 = 'res_152_32_8_sgd_005.pth'
    d4 = 'res_152_32_100_adam_betas_.pth'
    d5 = 'res_152_32_10_sgd_001_.pth'

    # vals = d1.split('_')
    # rm = training_model(small, get_model(vals[1]), vals[0])
    # o1 = get_optim(list(rm.model.parameters()), vals[4], learn=0.00001)
    # try:
    #     n1 = rm.train(d1, o1, int(vals[2]), int(vals[3]))
    #     m1 = test(n1, vals[1], arch=vals[0])
    #     cm.run(n1, m1, 'NN_data/hot_or_not_oct_23/', 32, vals[0])
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d1)

    vals = d2.split('_')
    rm = training_model(small, get_model(vals[1]), vals[0])
    o1 = get_optim(list(rm.model.parameters()), vals[4], learn=0.0001)
    try:
        n2 = rm.train(d2, o1, int(vals[2]), int(vals[3]))
        m2 = test(n2, vals[1], arch=vals[0])
        cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32, vals[0])
    except Exception as e:
        print(e)
        torch.save(rm.model.state_dict(), d2)








