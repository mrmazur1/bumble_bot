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

from simpleCNN import SimpleCNN, myTransform
from downloadImage import imageGetter
import torch
from selenium import webdriver
import cookieManager


class bumbleLoader:
    def __init__(self, url="https://bumble.com"):
        edge_driver_path = os.path.join(os.getcwd(), 'web_driver/msedgedriver.exe')
        edge_service = Service(edge_driver_path)
        self.driver = webdriver.Edge(service=edge_service)
        self.driver.get(url)
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load('model_test.pth'))
        self.model.eval()

    def load(self):
        cookieManager.load_cookies(self.driver) #load password stored in cookie
        self.close_messages()  # close annoying popups
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
        #time.sleep(5)
        self.close_messages()
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
            self.close_messages()
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
                self.moveFiles(f"tmp/", f"outputs/liked/{str(i)}/")
            else: #dislike
                self.dislike.click()
                self.moveFiles(f"tmp/", f"outputs/disliked/{str(i)}/")
            time.sleep(1)
        self.driver.quit()

    def moveFiles(self, source, destination):
        if os.path.exists(destination):
            os.rmdir(destination)
            os.makedirs(destination)
        for file in os.listdir(source):
            source_path = os.path.join(source, file)
            destination_path = os.path.join(destination, file)
            shutil.move(source_path, destination_path)

    def getImage(self, element):
        element.click()
        try:
            photo = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.XPATH,
                                                '//*[@id="main"]/div/div[1]/div[1]/div/div[2]/div/div/div[2]/article/div[1]/div/span/img')))
        finally:
            pass
        picture_Url = photo.get_attribute('src')
        path, filename = self.imget.getImage(picture_Url)
        self.driver.find_element(By.XPATH,
                                 '//*[@id="main"]/div/div[1]/div[1]/div/div[2]/div/div/div[2]/article/div[3]').click()
        return path, filename

    def rateImages(self):
        numImages, liked = 0, 0
        end = False
        extenders = self.driver.find_elements(By.TAG_NAME, 'span')
        for extender in extenders:
            if extender.get_attribute('class') != 'icon icon--size-m':
                 continue
            while not extender.is_displayed():
                if 'is-disabled' in self.down.get_attribute('class'):
                    end = True
                    break
                self.down.click()
                time.sleep(1)
            time.sleep(1)
            if end: break
            numImages+=1
            try:
                img_path, img_name = self.getImage(extender)
                pred = self.predict(img_path, img_name)
                if pred == 'hot':
                    liked+=1
            except Exception as e:
                pass
            self.down.click()
        #self.getImage()
        return liked > numImages//2

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
        # os.remove('img.jpg')

    def close_messages(self):
        iframes = self.driver.find_elements(By.TAG_NAME, 'iframe')
        for frame in iframes:
            # time.sleep(250/1000)
            self.driver.switch_to.frame(frame)
            inner = self.driver.find_elements(By.CSS_SELECTOR, 'button')
            for inVal in inner:
                if 'Continue' in inVal.get_attribute('title'):
                    inVal.click()
            self.driver.switch_to.parent_frame()

#TODO finish bumbleLoader