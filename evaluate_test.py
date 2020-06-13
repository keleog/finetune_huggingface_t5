from absl import app
from absl import flags

from src.evaluator import Evaluator

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "experiment_path", None, "Experiment output directory containing saved model and config file"
)
flags.DEFINE_string("src_test_path", None, "Path to source language test sentences")
flags.DEFINE_string("tgt_test_path", None, "Path to target language test sentences")
flags.DEFINE_boolean("save_as_pretrained", False, "If True, save the loaded model as pretrained")


def main(argv):

    # initialize evaluator and evaluate
    evaluator = Evaluator(
        FLAGS.experiment_path, FLAGS.src_test_path, FLAGS.tgt_test_path, FLAGS.save_as_pretrained,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    flags.mark_flags_as_required(["experiment_path", "src_test_path", "tgt_test_path"])
    app.run(main)
