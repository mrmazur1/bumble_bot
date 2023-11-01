import os
import pickle
import time

from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service


from bumbleLoader import bumbleLoader

def save_cookies():
    driver = bumbleLoader("https://bumble.com/get-started").driver
    print()
    cookies_list = driver.get_cookies()
    # for cook in cookies_list:
    #     #val = cook['name'] + " " + cook['value']
    #     print(cook)
    # print()
    try:
        element = WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.XPATH,
                                            '//*[@id="main"]/div/div[1]/main/div[2]/div/div/span/div[1]/article/div[1]/div[1]/article/div[1]/div/figure/div/div/span'))
        )
    finally:
        pass

    cookies_to_save = []
    for cookie in driver.get_cookies():
        # Check the attributes of each cookie and decide which to save
        if cookie['name'] == 'session' or cookie['name'] == 'HDR-X-User-id':
            cookies_to_save.append(cookie)
    pickle.dump(cookies_to_save, open("cookies.pkl", "wb"))
    driver.quit()

def load_cookies():
    edge_driver_path = os.path.join(os.getcwd(), 'web_driver/msedgedriver.exe')
    edge_service = Service(edge_driver_path)
    driver = webdriver.Edge(service=edge_service)
    driver.get("https://bumble.com")
    with open("cookies.pkl", "rb") as file:
        cookies = pickle.load(file)
    for cookie in cookies:
        driver.add_cookie(cookie)
    print()
    iframes = driver.find_elements(By.TAG_NAME, 'iframe')
    for frame in iframes:
        #time.sleep(250/1000)
        driver.switch_to.frame(frame)
        inner = driver.find_elements(By.CSS_SELECTOR, 'button')
        for inVal in inner:
            if 'message-button' in inVal.get_attribute('class'):
                inVal.click()
        driver.switch_to.parent_frame()
    try:
        element = WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.XPATH,
                                            '//*[@id="main"]/div[1]/div/div/div/div[2]/div/div[2]/div[1]/div/div[2]/a'))
        )
        driver.find_element(By.XPATH, '//*[@id="main"]/div[1]/div/div/div/div[2]/div/div[2]/div[1]/div/div[2]/a').click()
    except TimeoutException as e:
        print(e.msg)
        driver.quit()
    time.sleep(2)

