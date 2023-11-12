
from bumbleLoader import bumbleLoader
import traceback


if __name__ == "__main__":
    bl = bumbleLoader()
    try:
        bl.load()
        bl.start(num_swipes=10)
    except Exception as e:
        html = bl.driver.page_source
        bl.driver.save_screenshot("web_page_screenshot.png")
        bl.driver.quit()
        with open("web_page_source.html", "w", encoding="utf-8") as file:
            file.write(html)
        #print(e)
        traceback.print_exc()
    bl.driver.quit()
    print("done")