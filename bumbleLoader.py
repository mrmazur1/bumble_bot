import os
import time

from PIL import Image
from selenium.common import WebDriverException, TimeoutException
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import torch.nn.functional as F
import uuid
import shutil

from torchvision import models

from simpleCNN import SimpleCNN, myTransform
from downloadImage import imageGetter
import torch
import cookieManager
from selenium import webdriver

class bumbleLoader:
    def __init__(self, url="https://bumble.com"):
        edge_driver_path = os.path.join(os.getcwd(), 'web_driver/msedgedriver.exe')
        edge_service = Service(edge_driver_path)
        self.driver = webdriver.Edge(service=edge_service)
        self.driver.get(url)
        size = self.driver.get_window_size()
        if size['width'] < 850:
            self.driver.set_window_size(850, 1000)
        # self.model = SimpleCNN()
        self.model = models.resnet50()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load('model_4_4_10_res50.pth', map_location=torch.device('cpu')))
        self.model.eval()

    def load(self):
        cookieManager.load_cookies(self.driver) #load password stored in cookie
        self.close_popups()  # close annoying popups
        #login
        try:
            #wait for log in button
            element = WebDriverWait(self.driver, 60).until(
                EC.presence_of_element_located((By.XPATH,
                                                '//*[@id="main"]/div[1]/div/div/div/div[2]/div/div[2]/div[1]/div/div[2]/a'))
            )
            self.driver.find_element(By.XPATH,
                                '//*[@id="main"]/div[1]/div/div/div/div[2]/div/div[2]/div[1]/div/div[2]/a').click()
        except TimeoutException as e:
            print(e.msg)
            self.driver.quit()
        #logged in at this point
        time.sleep(2)
        self.close_popups()
        #wait for page to load
        try:
            photo = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[2]/div/div[1]/span')))
        finally:
            pass
        self.like = self.driver.find_element(By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[3]/div/div[1]/span')
        self.dislike = self.driver.find_element(By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[1]/div/div[1]/span')
        self.down = self.driver.find_element(By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[2]/div[2]')
        self.up = self.driver.find_element(By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[2]/div[1]')
        self.imget = imageGetter()

    def start(self, num_swipes=2):
        for i in range(num_swipes):
            time.sleep(1)
            self.close_popups()
            try:
                photo = WebDriverWait(self.driver, 30).until(
                        EC.presence_of_element_located((By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[2]/div/div[1]/span')))
            finally:
                pass
            name = self.driver.find_element(By.XPATH,
                                       '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[1]/div[1]/article/div[2]/section/header/h1/span[1]')
            print(name.text)
            pred = self.rateImages()
            if pred: #like
                self.like.click()
                #self.moveFiles(f"tmp/", f"outputs/liked/{str(i)}/")
            else: #dislike
                self.dislike.click()
                #self.moveFiles(f"tmp/", f"outputs/disliked/{str(i)}/")
            time.sleep(1)
        self.driver.quit()

    def moveFiles(self, source, destination):
        if os.path.exists(destination):
            shutil.rmtree(destination)
        os.makedirs(destination)
        for file in os.listdir(source):
            source_path = os.path.join(source, file)
            destination_path = os.path.join(destination, file)
            shutil.move(source_path, destination_path)

    def getImage(self, url):
        path, filename = self.imget.getImage(url)
        return path, filename

    def rateImages(self): #Take images and rate theit attractiveness
        numImages, liked = 0, 0
        pics = self.driver.find_elements(By.TAG_NAME, 'img')
        for pic in pics: #go through each profile pic
            #ignore all the other pics that are not the main focus like cht profile pics/my profile/icons/etc
            if pic.get_attribute('class') != 'media-box__picture-image':
                continue
            numImages+=1
            try:
                img_path, img_name = self.getImage(pic.get_attribute('src'))
                pred = self.predict(img_path, img_name)
                if pred == 'hot':
                    liked+=1
            except Exception as e:
                pass
        print(str(liked) + " "+ str(numImages))
        return liked > numImages//2

    #takes the image and makes a prediction based on the model
    def predict(self, picture_path, filename=None):
        # Load and preprocess your input data (e.g., an image)
        input_image = Image.open(picture_path+filename)
        input_tensor = myTransform().transform_input(input_image)
        input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

        with torch.no_grad():
            output = self.model(input_batch)

        class_labels = ['hot', 'not']

        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
        predicted_class = class_labels[predicted_class.item()]
        output = F.softmax(output, dim=1)
        print(f"filename: {filename}")
        print(f"predicted class: {predicted_class}")
        print(f"outputs: {output}\n")
        return predicted_class

    def close_popups(self):
        iframes = self.driver.find_elements(By.TAG_NAME, 'iframe')
        for frame in iframes:
            try:
                self.driver.switch_to.frame(frame)
                inner = self.driver.find_elements(By.CSS_SELECTOR, 'button')
                for inVal in inner:
                    if 'Continue' in inVal.get_attribute('title'):
                        inVal.click()
                self.driver.switch_to.parent_frame()
            except Exception as e:
                print(e)
                pass





