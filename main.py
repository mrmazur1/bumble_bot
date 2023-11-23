import os

import torch
from PIL import Image
from torchvision import models, transforms

from simpleCNN import Trainer, SimpleCNN, myTransform, Resnet_model, confusion_matrix_me, EarlyStopping

def get_resnet_model(model_type='resnet18'):
    available_models = {
        '18': models.resnet18(pretrained=True),
        '34': models.resnet34(pretrained=True),
        '50': models.resnet50(pretrained=True),
        '101': models.resnet101(pretrained=True),
        '152': models.resnet152(pretrained=True),
    }
    # Check if the specified model_type is in the available_models dictionary
    if model_type in available_models:
        # Instantiate the selected model and return it
        return available_models[model_type]
    else:
        # If the specified model_type is not found, raise an exception or return a default model
        raise ValueError(f"Invalid model type: {model_type}")

def bing(model, type='hot'):
    direc = f"NN_data/hot_or_not_oct_23/{type}/"
    cnt_hot, cnt_not = 0, 0
    tot = 0
    hot_img = []
    avg_hot, avg_not = 0,0
    for file in os.listdir(direc):
        # if tot > 20:
        #     break
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
        # img.show()
        preprocessed_image = transform(img).unsqueeze(0).to('cuda')
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
            #hot_img.append([filename, outputs])
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
    model = get_resnet_model(model)
    trainer = Resnet_model()
    trainer.train(name, 'NN_data/hot_or_not_oct_23', batch, pocs, model)
    return name

def test(name, model, type='hot'):
    model = get_resnet_model(model)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(name, map_location=torch.device('cuda')))
    model.to('cuda')
    model.eval()
    bing(model, type)
    return model

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform = myTransform().transform
    class_labels = ['Hot', 'Not']  # Replace with your actual class label

    cm = confusion_matrix_me()

    d1 = 'res_152_64_160_.pth'
    d2 = 'res_152_32_160_.pth'
    d3 = 'res_101_32_200.pth'
    d4 = 'res_34_64_200'

    vals = d1.split('_')
    rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model(vals[1]))
    try:
        n1 = rm.train(d1, int(vals[2]), int(vals[3]))
        m1 = test(n1, vals[1])
        cm.run(n1, m1, 'NN_data/hot_or_not_oct_23/', 32)
    except Exception as e:
        print(e)
        torch.save(rm.model.state_dict(), d1)

    vals = d2.split('_')
    rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model(vals[1]))
    try:
        n2 = rm.train(d2, int(vals[2]), int(vals[3]))
        m2 = test(n2, vals[1])
        cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    except Exception as e:
        print(e)
        torch.save(rm.model.state_dict(), d2)

    # vals = d3.split('_')
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model(vals[1]))
    # try:
    #     n1 = rm.train(d3, int(vals[2]), int(vals[3]))
    #     m1 = test(n1, vals[1])
    #     cm.run(n1, m1, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d3)
    #
    # vals = d4.split('_')
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model(vals[1]))
    # try:
    #     n2 = rm.train(d4, int(vals[2]), int(vals[3]))
    #     m2 = test(n2, vals[1])
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d4)









