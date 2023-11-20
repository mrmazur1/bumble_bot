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

    # model = test('res50_32_4', '50')
    # cm = confusion_matrix_me()
    # cm.run(model, 'NN_data/hot_or_not_oct_23/')

    cm = confusion_matrix_me()


    d1 = 'res50_128_30'
    d2 = 'res34_128_30'
    d3 = 'res18_128_30'
    d4 = 'res34_32_50'
    d5 = 'res18_64_50'
    d6 = 'res18_32_50'
    d7 = 'res101_64_50'
    d8 = 'res101_32_50'
    d9 = 'res152_64_50'
    d10 = 'res152_32_50'

    test(d1, '50', 'hot')
    test(d1, '50', 'not')
    test(d2, '34', 'hot')
    test(d2, '34', 'not')
    test(d3, '18', 'hot')
    test(d3, '18', 'not')
    test(d4, '34', 'hot')
    test(d4, '34', 'not')
    test(d5, '18', 'hot')
    test(d5, '18', 'not')
    test(d6, '18', 'hot')
    test(d6, '18', 'not')
    #
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('50'))
    # try:
    #     n1 = rm.train(d1, 128, 30)
    #     m1 = test(n1, '50')
    #     cm.run(n1, m1, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d1)
    #
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('34'))
    # try:
    #     n2 = rm.train(d2, 128, 30)
    #     m2 = test(n2, '34')
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d2)
    #
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('18'))
    # try:
    #     n1 = rm.train(d3, 128, 30)
    #     m1 = test(n1, '18')
    #     cm.run(n1, m1, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d3)

    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('34'))
    # try:
    #     n2 = rm.train(d4, 32, 50)
    #     m2 = test(n2, '34')
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d4)
    #
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('18'))
    # try:
    #     n2 = rm.train(d5, 64, 50)
    #     m2 = test(n2, '18')
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d5)
    #
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('18'))
    # try:
    #     n2 = rm.train(d6, 32, 50)
    #     m2 = test(n2, '18')
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d6)
    #
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('101'))
    # try:
    #     n2 = rm.train(d7, 64, 50)
    #     m2 = test(n2, '101')
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d7)

    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('101'))
    # try:
    #     n2 = rm.train(d8, 32, 50)
    #     m2 = test(n2, '101')
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d8)
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('152'))
    # try:
    #     n1 = rm.train(d9, 64, 50)
    #     m1 = test(n1, '152')
    #     cm.run(n1, m1, 'NN_data/hot_or_not_oct_23/', 64)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d9)
    #
    # rm = Resnet_model('NN_data/hot_or_not_oct_23', get_resnet_model('152'))
    # try:
    #     n2 = rm.train(d10, 32, 40)
    #     m2 = test(n2, '152')
    #     cm.run(n2, m2, 'NN_data/hot_or_not_oct_23/', 32)
    # except Exception as e:
    #     print(e)
    #     torch.save(rm.model.state_dict(), d10)








