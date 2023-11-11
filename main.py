import os

import torch
from PIL import Image
from IPython.display import display

from simpleCNN import Trainer, SimpleCNN, myTransform, Resnet_model
from bumbleLoader import bumbleLoader
import cookieManager
from scraper import scrapper


if __name__ == "__main__":
    #TODO keep this code cause it loads the main thing

    # bl = bumbleLoader()
    # bl.load()
    # bl.start()
    # cookieManager.run()


    trainer = Resnet_model()
    #naming is modelnum/batch size/num epochs/model type
    trainer.train('model_4_4_10_res152.pth', 'NN_data/hot_or_not_oct_23', 4, 10)

    # model = SimpleCNN()
    # model.load_state_dict(torch.load('model_test.pth'))
    # model.eval()
    #
    # transform = myTransform().transform
    # # model = model.to('cuda')
    #
    # class_labels = ['Hot', 'Not']  # Replace with your actual class labels
    #
    # direc = "NN_data/hot_or_not_oct_23/not/"
    # cnt = 0
    # for file in os.listdir(direc):
    #     if cnt > 4:
    #         break
    #     filename = os.fsdecode(file)
    #     img = Image.open(direc + filename)
    #     img.show()
    #     # display(img)
    #     preprocessed_image = transform(img).unsqueeze(0)
    #     # preprocessed_image = preprocessed_image.to('cuda')
    #     with torch.no_grad():
    #         logits = model(preprocessed_image)
    #         _, predicted_class_index = torch.max(logits, 1)
    #         predicted_label = class_labels[predicted_class_index.item()]
    #         outputs = torch.nn.functional.softmax(logits, dim=1)
    #
    #     print(f"Predicted Label: {predicted_label}")
    #     print(f"Predicted Probabilities: {outputs}")
    #     print()
    #     cnt+=1
    print("done")
