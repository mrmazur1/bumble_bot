import os
import time

from PIL import Image
from selenium.common import WebDriverException
from selenium.webdriver.chrome.service import Service
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


class bumbleLoader:
    def __init__(self, url="https://bumble.com/get-started"):
        edge_driver_path = os.path.join(os.getcwd(), 'web_driver/msedgedriver.exe')
        edge_service = Service(edge_driver_path)
        self.driver = webdriver.Edge(service=edge_service)
        self.driver.get(url)
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load('model_test.pth'))
        self.model.eval()

    def load(self):
        try:
            iframe = self.driver.find_element(By.ID, 'sp_message_iframe_810475')
            self.driver.switch_to.frame(iframe)
            button = self.driver.find_element(By.TAG_NAME, 'button')
            button.click()
        except WebDriverException:
            pass
        self.driver.switch_to.parent_frame()
        phone = self.driver.find_element(By.XPATH,
                                    '//*[@id="main"]/div/div[1]/div[2]/main/div/div[3]/form/div[3]/div/span/span/span')
        phone.click()
        phone = self.driver.find_element(By.XPATH, '//*[@id="phone"]')
        phone.send_keys('2018208304')
        self.driver.find_element(By.XPATH,
                            '//*[@id="main"]/div/div[1]/div[2]/main/div/div[3]/form/div[4]/button/span/span/span').click()

        try:
            element = WebDriverWait(self.driver, 60).until(
                EC.presence_of_element_located((By.XPATH,  '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[3]/div/div[1]/span'))
            )
        finally:
            pass

        self.like = self.driver.find_element(By.XPATH,
                                   '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[3]/div/div[1]/span')
        self.dislike = self.driver.find_element(By.XPATH,
                                      '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[2]/div/div[2]/div/div[1]/div/div[1]/span')
        self.imget = imageGetter()

    def start(self, num_swipes=2):
        for i in range(num_swipes):
            self.driver.find_element(By.XPATH,
                                '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[1]/div[1]/article/div[1]/div/figure/div/div/span').click()
            try:
                photo = WebDriverWait(self.driver, 30).until(
                        EC.presence_of_element_located((By.XPATH, '//*[@id="main"]/div/div[1]/div[1]/div/div[2]/div/div/div[2]/article/div[1]/div/span/img')))
            finally:
                pass
            name = self.driver.find_element(By.XPATH,
                                       '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[1]/div[1]/article/div[2]/section/header/h1/span[1]')
            picture_Url = photo.get_attribute('src')
            self.imget.getImage(picture_Url)
            print(name.text)
            self.driver.find_element(By.XPATH,
                                '//*[@id="main"]/div/div[1]/div[1]/div/div[2]/div/div/div[2]/article/div[3]').click()
            pred = self.predict('img.jpg', 'img.jpg')
            random_name = uuid.uuid4().hex
            if pred == 'hot':
                self.like.click()
                shutil.move('img.jpg', f"outputs/liked/{random_name}.jpg")
            else:
                self.dislike.click()
                shutil.move('img.jpg', f"outputs/disliked/{random_name}.jpg")
            time.sleep(1)
        self.driver.quit()

    def predict(self, picture_path, filename=None):
        # Load and preprocess your input data (e.g., an image)
        input_image = Image.open(picture_path)
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

#TODO finish bumbleLoader