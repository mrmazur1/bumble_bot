import pickle

def save_cookies(driver):
    # cookies_to_save = []
    # for cookie in driver.get_cookies():
    #     # Check the attributes of each cookie and decide which to save
    #     if cookie['name'] == 'session' or cookie['name'] == 'HDR-X-User-id':
    #         cookies_to_save.append(cookie)
    cookies_list = driver.get_cookies()
    pickle.dump(cookies_list, open("cookies.pkl", "wb"))

def load_cookies(driver):
    with open("cookies.pkl", "rb") as file:
        cookies = pickle.load(file)
    for cookie in cookies:
        driver.add_cookie(cookie)

