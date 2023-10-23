
from bumbleLoader import bumbleLoader
import cookieManager
from scraper import scrapper


if __name__ == "__main__":
    bl = bumbleLoader()
    bl.load()
    bl.start()
    cookieManager.run()
    print("done")
