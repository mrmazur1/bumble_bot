import os

import torch
from PIL import Image
from torchvision import models, transforms

from simpleCNN import Trainer, SimpleCNN, myTransform, Resnet_model, confusion_matrix_me, EarlyStopping

def get_resnet_model(model_type='resnet18'):
    available_models = {
        '18': models.resnet18(),
        '34': models.resnet34(),
        '50': models.resnet50(),
        '101': models.resnet101(),
        '152': models.resnet152(),
    }
    # Check if the specified model_type is in the available_models dictionary
    if model_type in available_models:
        # Instantiate the selected model and return it
        return available_models[model_type]
    else:
        # If the specified model_type is not found, raise an exception or return a default model
        raise ValueError(f"Invalid model type: {model_type}")

def bing(model):
    direc = "NN_data/hot_or_not_oct_23/hot/"
    cnt_hot, cnt_not = 0, 0
    tot = 0
    hot_img = []
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

        if predicted_label == 'Not':
            cnt_not += 1
        else:
            cnt_hot += 1
            hot_img.append([filename, outputs])
        # print(f"filename: {filename}")
        # print(f"class_index: {predicted_class_index.item()}")
        # print(f"Predicted Label: {predicted_label}")
        # print(f"Predicted Probabilities: {outputs}")
        # print()
        tot += 1
    print(f"tot: {tot}\nnot: {cnt_not}\nhot: {cnt_hot}\n")

def train(name, batch, pocs, model):
    print(f"name: {name}")
    model = get_resnet_model(model)
    trainer = Resnet_model()
    trainer.train(name, 'NN_data/hot_or_not_oct_23', batch, pocs, model)
    return name

def test(name, model):
    model = get_resnet_model(model)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(name, map_location=torch.device('cuda')))
    model.to('cuda')

    model.eval()
    bing(model)
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

    # model = test('res50_32_4', '50')
    # cm = confusion_matrix_me()
    # cm.run(model, 'NN_data/hot_or_not_oct_23/')

    cm = confusion_matrix_me()

    rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('50'))
    d1 = 'res50_64_100'
    d2 = 'res50_32_100'
    d3 = 'res34_32_50'
    d4 = 'res34_64_50'
    d5 = 'res18_32_50'
    d6 = 'res18_64_50'

    try:
        n1 = rm.train(d1, 64, 100)
        m1 = test(n1, '50')
        cm.run(n1, m1, 'NN_data/hot_or_not_oct_23/', 64)
    except Exception as e:
        print(e)
        torch.save(rm.model.state_dict(), d1)

    rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('50'))
    try:
        n2 = rm.train(d2, 32, 50)
        m2 = test(n2, '50')
        cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    except Exception as e:
        print(e)
        torch.save(rm.model.state_dict(), d2)

    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('34'))
    # try:
    #     n1 = rm.train(d3, 64, 50)
    #     m1 = test(n1, '50')
    #     cm.run(n1, m1, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d3)
    #
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('34'))
    # try:
    #     n2 = rm.train(d4, 32, 50)
    #     m2 = test(n2, '50')
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d4)
    #
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('18'))
    # try:
    #     n2 = rm.train(d5, 32, 50)
    #     m2 = test(n2, '50')
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d5)
    #
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('18'))
    # try:
    #     n2 = rm.train(d6, 32, 50)
    #     m2 = test(n2, '50')
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d6)








