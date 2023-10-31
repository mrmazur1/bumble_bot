import os

import torch
from PIL import Image
from IPython.display import display

from simpleCNN import Trainer, SimpleCNN, myTransform
from bumbleLoader import bumbleLoader
import cookieManager
from scraper import scrapper


if __name__ == "__main__":
    # bl = bumbleLoader()
    # bl.load()
    # bl.start(num_swipes=5)
    #cookieManager.run()
    cookieManager.load_cookies()
    print("done")