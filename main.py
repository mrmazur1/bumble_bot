from bumbleLoader import bumbleLoader
import traceback
import time

if __name__ == "__main__":
    count = 1
    exit_flag = False
    bl = bumbleLoader(modelType='152', modelPath='res_152_32_100.pth')
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
            bl.driver.quit()
            with open("web_page_source.html", "w", encoding="utf-8") as file:
                file.write(html)
                # print(e)
                traceback.print_exc()
            bl = bumbleLoader(modelType='152', modelPath='res_152_32_100.pth')
            bl.load()
            bl.start(val, numLikes=nlikes, numDislikes=ndislikes, num_swipes=count - val)
    print(f"ending after completing {bl.tracker} profiles with {bl.numLikes} likes and {bl.numDislikes} dislikes")
    bl.driver.quit()
