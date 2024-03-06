from bumbleLoader import bumbleLoader
import traceback
import time
from selenium.common.exceptions import TimeoutException
import os
import cookieManager

def save_cookie(driver):
    input("continue: ")
    cookieManager.save_cookies(driver)
    cookieManager.load_cookies(driver)


if __name__ == "__main__":
    #TODO make way to quit when constantly throwing exceptions
    count = int(input("how many profiles do you want to run: "))
    option = int(input("what do you want to do? 1 (save cookie), 2 (run normally)"))
    #count = 10
    exit_flag = False
    bl = bumbleLoader(modelType='201', modelPath='models/dense_201_64_70_adam_.pth', arch='dense')
    if option == 1:
        save_cookie(bl.driver)
    start = time.monotonic()
    while bl.tracker < count:
        curr = time.monotonic()
        if curr > start+3600: #1 hour max
            break
        try:
            bl.load()
            bl.start(0, num_swipes=count)
        except Exception as e:
            html = bl.driver.page_source
            bl.driver.save_screenshot("web_page_screenshot.png")
            val,nlikes, ndislikes = bl.tracker, bl.numLikes, bl.numDislikes
            print(f"restarting at count of {val}")
            bl.driver.get("https://www.google.com/")
            bl.driver.get("https://bumble.com")
            bl.load()
            bl.driver.quit()
            with open("web_page_source.html", "w", encoding="utf-8") as file:
                file.write(html)
                # print(e)
                traceback.print_exc()
            try:
                bl = bumbleLoader(modelType='201', modelPath='models/dense_201_64_70_adam_.pth', arch='dense')
                bl.load()
                bl.start(val, numLikes=nlikes, numDislikes=ndislikes, num_swipes=count - val)
            except TimeoutException as e:
                print("most likely came to end of profiles in area")
                break
    print(f"ending after completing {bl.tracker} profiles with {bl.numLikes} likes and {bl.numDislikes} dislikes")
    cookieManager.save_cookies(bl.driver)
    bl.driver.quit()

