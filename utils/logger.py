import logging
import os
import sys


def make_logger(name, save_dir, save_filename):
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt=DATE_FORMAT)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, save_filename + ".txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
