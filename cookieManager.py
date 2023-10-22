import pickle
from bumbleLoader import bumbleLoader

def run():
    driver = bumbleLoader("https://www.youtube.com").driver
    pickle.dump(driver.get_cookies(), open("cookies.pkl", "wb"))
