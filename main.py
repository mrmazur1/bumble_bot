import os

import torch
from PIL import Image
from IPython.display import display
from torchvision import models, transforms

from simpleCNN import Trainer, SimpleCNN, myTransform, Resnet_model, confusion_matrix_me

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


if __name__ == "__main__":
    # cm = confusion_matrix_me()
    # cm.run(model, 'NN_data/hot_or_not_oct_23/')
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

    n1 = train('res18_32_1', 32, 1, '18')
    test(n1, '18')
    n2 = train('res50_32_1', 32, 1, '50')
    test(n2, '50')
    n3 = train('res34_32_1', 32, 1, '34')
    test(n3, '34')






