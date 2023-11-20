import uuid
import os
import requests

class imageGetter():
    def __init__(self):
        pass
    def getImage(self, url, idx):
        data = requests.get(url).content
        random_name = uuid.uuid4().hex
        direc = f"tmp/{idx}/"
        if not os.path.exists(direc):
            os.mkdir(direc)
        f = open(f'tmp/{idx}/{random_name}.jpg', 'wb')
        f.write(data)
        f.close()
        #img = Image.open(f'{random_name}.jpg')
        return f'tmp/{idx}/', f'{random_name}.jpg'