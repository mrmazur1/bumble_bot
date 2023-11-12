import uuid

import requests
from PIL import Image

class imageGetter():
    def __init__(self):
        pass
    def getImage(self, url):
        data = requests.get(url).content
        random_name = uuid.uuid4().hex
        f = open(f'tmp/{random_name}.jpg', 'wb')
        f.write(data)
        f.close()
        #img = Image.open(f'{random_name}.jpg')
        return 'tmp/', f'{random_name}.jpg'