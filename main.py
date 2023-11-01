import os

import torch
from PIL import Image
from IPython.display import display

from simpleCNN import Trainer, SimpleCNN, myTransform
from bumbleLoader import bumbleLoader
from scraper import scrapper


if __name__ == "__main__":
    bl = bumbleLoader()
    bl.load()
    bl.start(num_swipes=2)
    #cookieManager.run()
    print("done")