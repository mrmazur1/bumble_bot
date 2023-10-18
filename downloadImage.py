import requests
from PIL import Image

class imageGetter():
    def __init__(self):
        pass
    def getImage(self, url):
        data = requests.get(url).content
        f = open('img.jpg', 'wb')
        f.write(data)
        f.close()
        img = Image.open('img.jpg')
        return img