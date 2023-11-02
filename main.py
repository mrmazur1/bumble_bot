import os

import torch
from PIL import Image
from IPython.display import display

from simpleCNN import Trainer, SimpleCNN, myTransform
from bumbleLoader import bumbleLoader
from scraper import scrapper

import traceback
from selenium import webdriver



if __name__ == "__main__":
    bl = bumbleLoader()
    try:

        bl.load()
        bl.start(num_swipes=4)
    except Exception as e:
        html = bl.driver.page_source
        bl.driver.save_screenshot("web_page_screenshot.png")
        bl.driver.quit()
        with open("web_page_source.html", "w", encoding="utf-8") as file:
            file.write(html)

        print(e)
        traceback.print_exc()
    #cookieManager.run()
    print("done")