
from bumbleLoader import bumbleLoader
import cookieManager
from scraper import scrapper


if __name__ == "__main__":
    # bl = bumbleLoader()
    # bl.load()
    # bl.start()
    #cookieManager.run()
    scr = scrapper()
    url = input("url: ")
    num = input("pics: ")
    scr.scrape(url, int(num))
    #print("done")
