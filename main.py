import os

import torch
from PIL import Image
from IPython.display import display
from torchvision import models, transforms

from simpleCNN import Trainer, SimpleCNN, myTransform, Resnet_model
from bumbleLoader import bumbleLoader
import cookieManager
from scraper import scrapper


if __name__ == "__main__":

    # trainer = Resnet_model()
    # #naming is modelnum/batch size/num epochs/model type
    # trainer.train('tester_32_3_res50.pth', 'NN_data/hot_or_not_oct_23', 32, 3)

    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('tester_32_3_res50.pth', map_location=torch.device('cuda')))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform = myTransform().transform
    # model = model.to('cuda')

    class_labels = ['Hot', 'Not']  # Replace with your actual class labels

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
                os.remove(direc+filename)
                continue
        except Exception as e:
            os.remove(direc+filename)
            continue
        #img.show()
        preprocessed_image = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(preprocessed_image)
            _, predicted_class_index = torch.max(logits, 1)
            predicted_label = class_labels[predicted_class_index.item()]
            outputs = torch.nn.functional.softmax(logits, dim=1)

        if predicted_label == 'Not':
            cnt_not +=1
        else:
            cnt_hot+=1
            hot_img.append([filename, outputs])
        print(f"filename: {filename}")
        print(f"class_index: {predicted_class_index.item()}")
        print(f"Predicted Label: {predicted_label}")
        print(f"Predicted Probabilities: {outputs}")
        print()
        tot+=1
    print(f"tot: {tot}\nnot: {cnt_not}\nhot: {cnt_hot}")

    print(hot_img)
