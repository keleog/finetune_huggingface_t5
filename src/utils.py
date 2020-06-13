import logging
from logging import Logger
from typing import Generator

import yaml
from sacrebleu import raw_corpus_bleu


def load_config(filename: str):
    """
    loads yaml configuration file.
    """
    conf_file = yaml.full_load(open(filename, "r"))
    return conf_file


def calculate_bleu_score(hyps: list, refs: list) -> float:
    """
    calculates bleu score.
    """
    assert len(refs) == len(hyps), "no of hypothesis and references sentences must be same length"
    bleu = raw_corpus_bleu(hyps, [refs])
    return bleu.score


def chunks(lst: list, n: int) -> Generator[list, None, None]:
    """
    Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def create_logger(log_file: str) -> Logger:
    """
    Create logger for logging the experiment process.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(file_handler)
    file_handler.setFormatter(formatter)

    return logger
