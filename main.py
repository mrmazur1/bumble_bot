from bumbleLoader import bumbleLoader
import traceback
import time
import os
import shutil

def move():
    direc = 'outputs/'
    for root, dirs, files in os.walk(direc):
        for file in files:
            name = os.fsdecode(file)
            print(file)
            file_path = os.path.join(root, file)
            print(file_path)
            #shutil.move(file_path, 'C:/Users/Mazur/Desktop/bumble_bot/bumble_bot/images/' + name)

if __name__ == "__main__":
    count = 300
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

