
from bumbleLoader import bumbleLoader
import traceback


if __name__ == "__main__":
    val, count = 0, 1
    exit_flag = False
    bl = bumbleLoader(modelType='101', modelPath='res101_64_50')
    while bl.tracker < count:
        try:
            bl.load()
            bl.start(val, num_swipes=count)
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