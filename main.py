from simpleCNN import Trainer
from bumbleLoader import bumbleLoader
import cookieManager
from scraper import scrapper


if __name__ == "__main__":
    #TODO keep this code cause it loads the main thing

    # bl = bumbleLoader()
    # bl.load()
    # bl.start()
    # cookieManager.run()

    trainer = Trainer()
    trainer.train_model('model_test.pth', 'NN_data/hot_or_not_oct_23', 32, 2)
    print("done")
