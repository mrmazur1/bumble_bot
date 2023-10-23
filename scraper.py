import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import uuid
import requests

class scrapper:

    def scrape(self, url, nums):
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        edge_driver_path = os.path.join(os.getcwd(), 'web_driver/msedgedriver.exe')
        edge_service = Service(edge_driver_path)
        driver = webdriver.Edge(service=edge_service)
        driver.get(url)
        time.sleep(3)
        try:
            driver.find_element(By.XPATH, '//*[@id="vs_images"]/div/div/ul/li[1]/div/div/div[1]/div/a/img').click()
            img = driver.find_element(By.XPATH, '/html/body/div[2]/div/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div/img')
            direc = "ad/"
            count = 0
            urls = set()
            end = time.time()+60
            while count < nums:
                #curr =
                try:
                    # if curr > end:
                    #     break
                    url = img.get_attribute('src')
                    if img.get_attribute('src') is not None and img.size['height'] > 192 and img.size[
                        'width'] > 192 and url not in urls:
                        data = requests.get(url).content
                        unique_filename = str(uuid.uuid4())
                        full_path = os.path.join(direc, unique_filename)
                        f = open(f'{full_path}.jpg', 'wb')
                        f.write(data)
                        f.close()
                        urls.add(url)
                        print(count)
                        count += 1
                    next = WebDriverWait(driver, 3).until(
                        EC.presence_of_element_located((By.XPATH, '//*[@id="navr"]/span'))
                    )
                    next.click()
                    img = WebDriverWait(driver, 3).until(
                        EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/div/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div/img'))
                    )
                except Exception as e:
                    driver.find_element(By.XPATH,
                                        '//*[@id="vs_images"]/div/div/ul/li[1]/div/div/div[1]/div/a/img').click()
                    img = WebDriverWait(driver, 3).until(
                        EC.presence_of_element_located(
                            (By.XPATH, '/html/body/div[2]/div/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div/img'))
                    )
                    pass
        except Exception as e:
            print(e)

