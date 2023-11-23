import os
import time

from PIL import Image
from selenium.common import TimeoutException
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import torch.nn.functional as F
import shutil

from torchvision import models

from simpleCNN import myTransform
from downloadImage import imageGetter
import torch
import cookieManager
from selenium import webdriver


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

class bumbleLoader:
    def __init__(self, url="https://bumble.com", modelType='18', modelPath='res18_32_50'):
        edge_driver_path = os.path.join(os.getcwd(), 'web_driver/msedgedriver.exe')
        edge_service = Service(edge_driver_path)
        self.driver = webdriver.Edge(service=edge_service)
        self.driver.get(url)
        size = self.driver.get_window_size()
        if size['width'] < 850:
            self.driver.set_window_size(850, 1000)
        # self.model = SimpleCNN()
        self.model = get_resnet_model(modelType)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
        self.model.eval()
        self.tracker, self.numLikes, self.numDislikes = 0,0,0

        #remove files at beginning
        shutil.rmtree('outputs')  # clear any previous data
        os.mkdir('outputs/')
        shutil.rmtree('tmp')
        os.mkdir('tmp/')

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
            photo = WebDriverWait(self.driver, 3).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[2]/div/div[1]/span')))
        finally:
            pass
        self.like = self.driver.find_element(By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[3]/div/div[1]/span')
        self.dislike = self.driver.find_element(By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[1]/div/div[1]/span')
        self.down = self.driver.find_element(By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[2]/div[2]')
        self.up = self.driver.find_element(By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[2]/div[1]')
        self.imget = imageGetter()

    def start(self, tracker, numLikes=0, numDislikes=0,num_swipes=2):
        self.tracker, self.numLikes, self.numDislikes = tracker, numLikes, numDislikes
        for i in range(num_swipes):
            self.tracker+=1
            if self.tracker > num_swipes: return
            self.idx = i
            time.sleep(1)
            self.close_popups()
            try: #this checks if a profile pops up
                photo = WebDriverWait(self.driver, 3).until(
                        EC.presence_of_element_located((By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[2]/div/div[1]/span')))
            finally:
                pass
            name = self.driver.find_element(By.XPATH,
                                       '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[1]/div[1]/article/div[2]/section/header/h1/span[1]')
            print(name.text)
            pred = self.rateImages()
            if pred: #like
                self.like.click()
                self.numLikes+=1
                self.moveFiles(f"tmp/{self.idx}/", f"outputs/liked/{str(i)}/")
            else: #dislike
                self.dislike.click()
                self.numDislikes+=1
                self.moveFiles(f"tmp/{self.idx}/", f"outputs/disliked/{str(i)}/")
            time.sleep(1)
        return

    def moveFiles(self, source, destination): #moves files from tmp to either the like or dislike folder
        if os.path.exists(destination):
            shutil.rmtree(destination)
        os.makedirs(destination)
        for file in os.listdir(source):
            source_path = os.path.join(source, file)
            destination_path = os.path.join(destination, file)
            shutil.copy(source_path, destination_path)

    def getImage(self, url):
        path, filename = self.imget.getImage(url, self.idx)
        return path, filename

    def rateImages(self): #Take images and rate theit attractiveness
        numImages, liked = 0, 0
        pics = self.driver.find_elements(By.TAG_NAME, 'img')
        for pic in pics: #go through each profile pic
            #ignore all the other pics that are not the main focus like cht profile pics/my profile/icons/etc
            if pic.get_attribute('class') != 'media-box__picture-image':
                continue
            numImages += 1
            try:
                img_path, img_name = self.getImage(pic.get_attribute('src'))
                pred_hot, pred_not = self.predict(img_path, img_name)
                if pred_hot > pred_not:
                    liked += 1
            except Exception as e:
                print(e)
                pass
        print(str(liked) + " " + str(numImages))
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
        val = output.cpu().numpy()
        return val[0,0], val[0,1] #values for like and dislike as percentages

    #close any popup notifications. these are easy to deal with while other stuff
    # will just have the program restart the build if an expected event pops up
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





