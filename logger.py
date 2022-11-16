import os
import logging

class Logger():
    def __init__(self, exp_path) -> None:

        logging.basicConfig(filename=os.path.join(exp_path, 'training.log'),
                            format='%(asctime)s %(levelname)s %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')

    def log(self, text):
        logging.info(text)