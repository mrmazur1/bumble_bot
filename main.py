from bumbleLoader import bumbleLoader
import traceback
import time


if __name__ == "__main__":
    count = 30
    exit_flag = False
    bl = bumbleLoader(modelType='101', modelPath='res_101_32_200')
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
            val = bl.tracker
            bl.driver.quit()
            with open("web_page_source.html", "w", encoding="utf-8") as file:
                file.write(html)
                # print(e)
                traceback.print_exc()
            bl = bumbleLoader(modelType='101', modelPath='res101_64_50')
            bl.load()
            bl.start(val, num_swipes=count - val)
    bl.driver.quit()
    print("done")