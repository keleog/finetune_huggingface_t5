import logging
import os

import torch
from dataset import MonoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from utils import calculate_bleu_score
from utils import load_config

from finetune_t5 import EXPERIMENT_CONFIG_NAME

logging.root.setLevel(logging.NOTSET)


class Evaluator:
    """
    This is an evaluator class for evaluating a t5 huggingface model on a parallel test set
    Attributes:
        model_config: model configuration
        train_config: training configuration
        data_config: data paths configuration
        experiment_path: path where experiment output will be dumped
        tokenizer: tokenizer
        src_test_generator: data generator for source test sentences
        tgt_test: list of target test sentences
        device: device where experiment will happen (gpu or cpu)
    """

    def __init__(
        self,
        experiment_path: str,
        src_test_path: str,
        tgt_test_path: str,
        save_as_pretrained: bool = False,
    ):

        self._check_inputs(experiment_path)

        config = load_config(os.path.join(experiment_path, EXPERIMENT_CONFIG_NAME))
        self.model_config = config["model"]
        self.train_config = config["training"]
        self.data_config = config["data"]
        self.experiment_path = experiment_path

        if self.model_config["tokenizer_path"]:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_config["tokenizer_path"])
        else:
            logging.warning(
                f"No tokenizer path inputed, using {self.model_config['model_size']} default pretrained tokenizer"
            )
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_config["model_size"])

        self.device = torch.device(
            "cuda" if self.train_config["use_cuda"] and torch.cuda.is_available() else "cpu"
        )
        self._load_model(experiment_path, save_as_pretrained)
        self.src_test_generator, self.tgt_test = self._create_datasets(src_test_path, tgt_test_path)

    def _load_model(self, experiment_path: str, save_as_pretrained: bool) -> None:
        """
        Loads trained model weights and saves as a huggingface pretrained model if specified.
        """
        logging.info("Loading model...")
        model_config = {
            "early_stopping": self.train_config["early_stopping"],
            "max_length": self.train_config["max_output_length"],
            "num_beams": self.train_config["beam_size"],
            "prefix": self.data_config["src_prefix"],
            "vocab_size": self.tokenizer.vocab_size,
        }
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_config["model_size"])
        self.model.config.update(model_config)

        checkpoint = torch.load(os.path.join(experiment_path, "best_model.pt"))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        if save_as_pretrained:
            pretrained_path = os.path.join(experiment_path, "best_model.bin")
            self.model.save_pretrained(pretrained_path)
            logging.info(
                f"Loaded model saved as pretrained model in path: {pretrained_path} ! Can now be loaded with: 'model.from_pretrained(path)' "
            )

    def _create_datasets(self, src_test_path: str, tgt_test_path: str) -> tuple:
        """
        Creates source test data generator and target reference data.
        """
        src_test = [
            self.model.config.prefix + text.strip() + " </s>" for text in list(open(src_test_path))
        ]
        tgt_test = [text.strip() for text in list(open(tgt_test_path))]

        assert len(src_test) == len(
            tgt_test
        ), "Source and Target data must have the same number of sentences"
        logging.info(f"Evaluating on datasets of {len(src_test)} sentences each...")

        src_test_dict = self.tokenizer.batch_encode_plus(
            src_test,
            max_length=self.train_config["max_output_length"],
            return_tensors="pt",
            pad_to_max_length=True,
        )
        params = {
            "batch_size": self.train_config["batch_size"],
            "shuffle": False,
            "num_workers": self.train_config["num_workers_data_gen"],
        }
        input_test_ids = src_test_dict["input_ids"]

        src_test_generator = DataLoader(MonoDataset(input_test_ids), **params)

        all_data = (src_test_generator, tgt_test)
        return all_data

    def evaluate(self) -> None:
        """
        Evaluate test data according to bleu score.
        """
        logging.info(f"Evaluating model with this configuration: \n {self.model.config}")

        # generate predictions and calculate bleu score
        hyps = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.src_test_generator):
                batch = batch.to(self.device)
                translations = self.model.generate(input_ids=batch)
                decoded = [
                    self.tokenizer.decode(
                        translation, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    for translation in translations
                ]
                hyps = hyps + decoded
        bleu_score = calculate_bleu_score(hyps, self.tgt_test)
        logging.info(f"BLEU score on test data is: {bleu_score:.2f}")

        # write hypothesis to file
        hyps_path = os.path.join(self.experiment_path, f"model_test_hyps_bleu:{bleu_score:.2f}.txt")

        with open(hyps_path, "w") as file:
            for sent in hyps:
                file.write(sent + " \n")
        logging.info(f"Model hypothesis saved in {hyps_path}")

    def _check_inputs(self, experiment_path: str) -> None:
        """
        check that input experiment path contains files needed for evaluation.
        """
        # check there is a single model named best_mode.pt in path
        assert (
            len(list(filter(lambda x: x == "best_model.pt", os.listdir(experiment_path)))) == 1
        ), "A single model file must exist and be named as 'best_model.pt' "

        # check config file is in path
        assert os.path.isfile(
            os.path.join(experiment_path, EXPERIMENT_CONFIG_NAME)
        ), f"Configuration file must exist in experiment path as {EXPERIMENT_CONFIG_NAME}"
