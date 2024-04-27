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

#from simpleCNN import myTransform
import myData
from downloadImage import imageGetter
import torch
import cookieManager
from selenium import webdriver

def get_model(model_type='resnet18'):
    available_models = {
        '18': models.resnet18(pretrained=True),
        '34': models.resnet34(pretrained=True),
        '50': models.resnet50(pretrained=True),
        '101': models.resnet101(pretrained=True),
        '152': models.resnet152(pretrained=True),
        '121': models.densenet121(pretrained=True),
        '161': models.densenet161(pretrained=True),
        '169': models.densenet169(pretrained=True),
        '201': models.densenet201(pretrained=True),
        'google': models.googlenet(pretrained=True), #only one
        '11': models.vgg11(pretrained=True),
        '19':models.vgg19(pretrained=True),
        '13': models.vgg13(pretrained=True),
        '16':models.vgg16(pretrained=True),
        'alex':models.alexnet(pretrained=True)
    }
    if model_type in available_models:
        return available_models[model_type]
    else:
        raise ValueError(f"Invalid model type: {model_type}")

class bumbleLoader:
    def __init__(self, url="https://bumble.com", modelType='18', modelPath='res18_32_50', arch='res'):
        """
        connect to bumble, load model, and clear old data
        :param url: url to site
        :param modelType: model type like 'res18' or 'densenet201'
        :param modelPath: path to model dictionary to load
        :param arch: specifies which type of model architecture to use like 'res' or' dense' or 'google'
        """
        edge_driver_path = os.path.join(os.getcwd(), 'web_driver/msedgedriver.exe')
        edge_service = Service(edge_driver_path)
        self.driver = webdriver.Edge(service=edge_service)
        self.driver.get(url)
        size = self.driver.get_window_size()
        if size['width'] < 850:
            self.driver.set_window_size(850, 1000)
        # self.model = SimpleCNN()
        self.model = get_model(modelType)
        if arch == 'res':
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        if arch == 'dense':
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, 2)
        if arch == 'vgg':
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = torch.nn.Linear(num_ftrs, 2)
        if arch == 'google':
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, 2)
        if arch == 'alex':
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, 2)
        #self.model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
        self.model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
        self.model.eval()
        self.tracker, self.numLikes, self.numDislikes = 0,0,0
        data = myData.get_architecture(arch)
        self.transform = data.transform
        self.class_labels = data.class_labels
        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.imget = imageGetter()

        #remove files at beginning
        shutil.rmtree('outputs')  # clear any previous data
        os.mkdir('outputs/')
        shutil.rmtree('tmp')
        os.mkdir('tmp/')

    def load(self):
        """
        this loads up the bumbleloader by logging in and making sure it comes in contact with a profile
        """
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
        time.sleep(2) #dont move it will break program
        self.close_popups()
        #wait for page to load
        try:
            photo = WebDriverWait(self.driver, 3).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[2]/div/div[1]/span')))
        finally:
            pass
        # self.like = self.driver.find_element(By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[3]/div/div[1]/span')
        # self.dislike = self.driver.find_element(By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[1]/div/div[1]/span')
        # self.down = self.driver.find_element(By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[2]/div[2]')
        # self.up = self.driver.find_element(By.XPATH, '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[2]/div[1]')


    def start(self, tracker, numLikes=0, numDislikes=0,num_swipes=2):
        """
        :param tracker: tracks total number of profiles analyzed
        :param numLikes: tracks the amount of profiles the liked
        :param numDislikes: tracks number of profiles disliked
        :param num_swipes: the number of swipes to go until the goal is reached.
        """

        self.like = self.driver.find_element(By.XPATH,
                                             '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[3]/div/div[1]/span')
        self.dislike = self.driver.find_element(By.XPATH,
                                                '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[1]/div/div[1]/span')
        self.down = self.driver.find_element(By.XPATH,
                                             '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[2]/div[2]')
        self.up = self.driver.find_element(By.XPATH,
                                           '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[2]/div[1]')

        self.tracker, self.numLikes, self.numDislikes = tracker, numLikes, numDislikes
        for i in range(num_swipes):
            print(f"profile number: {self.tracker}")
            self.tracker+=1
            if self.tracker > num_swipes: return
            self.idx = i
            time.sleep(1) #dont change it will break program
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
            time.sleep(1) #dont move it will break program

    def moveFiles(self, source, destination): #moves files from tmp to either the like or dislike folder
        """
        :param source: source of file and context root
        :param destination: new destination to move file to
        """
        if os.path.exists(destination):
            shutil.rmtree(destination)
        os.makedirs(destination)
        for file in os.listdir(source):
            source_path = os.path.join(source, file)
            destination_path = os.path.join(destination, file)
            shutil.copy(source_path, destination_path)

    def getImage(self, url):
        """
        uses imageGetter to download the source image using the image url
        :param url: source image url
        :return: return the contect root and the filename to the picture
        """
        path, filename = self.imget.getImage(url, self.idx)
        return path, filename

    def rateImages(self):
        """
        this method rates images in the profiles
        :return: return true if most of the pics are hot else false if >=50% are labeled not
        """
        numImages, liked = 0, 0
        pics = self.driver.find_elements(By.TAG_NAME, 'img')
        for pic in pics: #go through each profile pic
            #ignore all the other pics that are not the main focus like cht profile pics/my profile/icons/etc
            if pic.get_attribute('class') != 'media-box__picture-image':
                continue
            numImages+=1
            try:
                img_path, img_name = self.getImage(pic.get_attribute('src'))
                pred_hot, pred_not = self.predict(img_path, numImages, img_name)
                if pred_hot == -1:
                    continue
                if pred_hot > pred_not:
                    liked += 1
            except Exception as e:
                print(e)
                pass
        print(str(liked) + " " + str(numImages))
        return liked > numImages//2

    def detect_human(self, picture_path, filename=None):
        """
        :param picture_path: context root path of the file
        :param filename: the name of the file without context root
        :return: returns
        """
        img = Image.open(picture_path+filename)
        results = self.yolo(img)
        detected_objects = results.pandas().xyxy[0]
        print(f"person detected in photo: {'person' in detected_objects.values}")
        return 'person' in detected_objects.values

    def predict(self, picture_path, numImages, filename=None):
        """
        it makes a prediciton on the profiles attractiveness if a person is detected
        :param picture_path: path to picture to analyze
        :param numImages: number of images being analyzed
        :param filename: file name to picture
        :return: returns the predicted values for hot or not returning -1, -1
        if no person is detected
         """
        # Load and preprocess your input data (e.g., an image)
        if not self.detect_human(picture_path, filename):
            numImages-=1
            print()
            return -1, -1
        input_image = Image.open(picture_path+filename)
        input_tensor = self.transform(input_image)
        input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

        with torch.no_grad():
            output = self.model(input_batch)

        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
        predicted_class = self.class_labels[predicted_class.item()]
        output = F.softmax(output, dim=1)
        print(f"filename: {filename}")
        print(f"predicted class: {predicted_class}")
        print(f"outputs: {output}\n")
        val = output.cpu().numpy()
        return val[0,0], val[0,1] #values for like and dislike as percentages


    def close_popups(self):
        #TODO: look into making this more efficient
        """
        closes annoying popups in the app that block clicking of profiles
        """
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





