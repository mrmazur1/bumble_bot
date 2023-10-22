import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait


class scrapper:

    def scrape(self, url):
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        edge_driver_path = os.path.join(os.getcwd(), 'web_driver/msedgedriver.exe')
        edge_service = Service(edge_driver_path)
        driver = webdriver.Edge(service=edge_service)
        driver.get(url)
        time.sleep(4)
        try:
            list_items = driver.find_elements(By.TAG_NAME, 'img')
            direc = "ad/"
            count = 0
            for li in list_items:
                if li.get_attribute('src') is not None and li.size['height'] > 0 and li.size['width'] > 0:
                    li.screenshot(direc+str(count)+'.png')
                    count+=1
        except Exception as e:
            print("An error occurred:", e)
        finally:
            driver.quit()
