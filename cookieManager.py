import pickle

def save_cookies(driver):
    cookies_to_save = []
    for cookie in driver.get_cookies():
        # Check the attributes of each cookie and decide which to save
        if cookie['name'] == 'session' or cookie['name'] == 'HDR-X-User-id':
            cookies_to_save.append(cookie)
    #cookies_list = driver.get_cookies()
    pickle.dump(cookies_to_save, open("cookies_new.pkl", "wb"))

def load_cookies(driver, desk=False):
    # if desk:
    #     with open("cookies_dek.pkl", "rb") as file:
    #         cookies = pickle.load(file)
    # else:
    with open("cookies_new.pkl", "rb") as file:
            cookies = pickle.load(file)
    for cookie in cookies:
        driver.add_cookie(cookie)

