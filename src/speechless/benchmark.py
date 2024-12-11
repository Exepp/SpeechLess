from enum import Enum
import re
from rouge import Rouge
from logging import Logger
from argparse import ArgumentParser

from speechless.processing.tokenization import EditToken
from speechless.processing.analysis.tfidf import TfidfAnalysis
from speechless.utils.logging import NULL_LOGGER
from speechless.utils.cli import cli_subcommand


class TokenGranularity(Enum):
    WORD = 0
    SENTENCE = 1

# dataset format: %[bad sentence%], %(bad_word%) 
class Benchmark:

    def __init__(self, dataset_file : str, method : str, granularity : TokenGranularity, logger: Logger = NULL_LOGGER):
        self.method = method
        self.granularity = granularity
        with open(dataset_file) as f:
            self.data = f.read().replace('\n', '')
            self.data = re.sub(' +', ' ', self.data)

    def run(self):
        expected_labels, gold_text = self._parse_dataset()
        self._process_data_to_classify()

        tokens = self._tokenize(self.data)

        # TODO: extend for more methods
        tfidf = TfidfAnalysis("text8", 0.1)
        float_labels = tfidf.score_transcription(tokens)
        actual_labels = []
        for label in float_labels:
            if label > 0.5:
                actual_labels.append(0)
            else:
                actual_labels.append(1)

        actual_text = self._get_text_from_labels(tokens, actual_labels)

        print(actual_text)

        rouge_res, labels_res = self._get_results(expected_labels, actual_labels, gold_text, actual_text)
        print("Rouge result: ")
        print(rouge_res)
        print("Comparing labels result: ")
        print(labels_res)

    def _get_text_from_labels(self, tokens, labels):
        text = ""
        assert len(tokens) == len(labels)
        for i in range(len(tokens)):
            if labels[i] == 1:
                text += " " + tokens[i].text
        return text
        
    def _label_text(self, erase_regex, bad_text_regex):
        text_to_label = re.sub(erase_regex, '', self.data)
        text_to_label = re.sub(' +', ' ', text_to_label)
        ranges = re.split(bad_text_regex, text_to_label)

        labels = []
        good_part = True
        for i in ranges:
            [labels.append(int(good_part)) for x in range(len(self._tokenize(i)))]
            good_part = not good_part

        return labels

    def _tokenize(self, data : str):
        return [EditToken(i, 0, 0.1) for i in data.split()]

    def _process_data_to_classify(self):
        self.data = re.sub(r'%\[|%\]|%\(|%\)', '', self.data)

    def _parse_dataset(self):
        if self.granularity == TokenGranularity.WORD:
            labels =  self._label_text(r'%\[|%\]', r'%\(|%\)')
            gold_text = re.sub(r'%\(.*?%\)|%\[|%\]', '', self.data)
        elif self.granularity == TokenGranularity.SENTENCE:
            labels = self._label_text(r'%\(|%\)', r'%\[|%\]')
            gold_text = re.sub(r'%\[.*?%\]|%\(.*?%\)', '', self.data)
        gold_text = re.sub(' +', ' ', gold_text)
        return labels, gold_text

    def _get_results(self, expected_labels, actual_labels, gold_text, actual_text):
        rouge_res = self._compare_rouge(gold_text, actual_text)
        labels_res = self._compare_labels(expected_labels, actual_labels)

        return rouge_res, labels_res

    def _compare_labels(self, expected_labels, actual_labels):
        total = 0
        correct = 0

        assert len(expected_labels) == len(actual_labels)
        for i in range(len(expected_labels)):
            total += 1
            if expected_labels[i] == actual_labels[i]:
                correct += 1
        return correct * 100 / total

    def _compare_rouge(self, gold_text, actual_text):
        rouge = Rouge()
        return rouge.get_scores(actual_text, gold_text)
 

############################################### CLI ################################################


@cli_subcommand
class CLI:

  COMMAND = 'benchmark'
  DESCRIPTION = 'Benchmarks specified methods'
  ARG_SRC = 'src'
  ARG_METHOD = 'method'
  ARG_GRANULARITY = 'granularity'
  DEFAULT_ARGS = {
      ARG_METHOD: 'tfidf',
      ARG_GRANULARITY: 'sentence'
  }

  @staticmethod
  def setup_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Sets up a CLI argument parser for this submodule

    Returns:
        ArgumentParser: Configured parser
    """
    parser.description = CLI.DESCRIPTION
    parser.add_argument(CLI.ARG_SRC,
                        help='Path of the file with dataset',
                        type=str,
                        action='store')
    parser.add_argument(CLI.ARG_METHOD,
                        help='Method to benchmark',
                        type=str,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_METHOD])
    parser.add_argument(CLI.ARG_GRANULARITY,
                        help='Granularity of method (word or sentence)',
                        type=str,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_GRANULARITY])
    parser.set_defaults(run=CLI.run_submodule)
    return parser

  @staticmethod
  def run_submodule(args: object, logger: Logger) -> None:
    """Runs this submodule

    Args:
        args (object): Arguments of this submodule (defined in setup_arg_parser)
        logger (Logger): Logger for messages
    """
    args = args.__dict__

    granularity = TokenGranularity.SENTENCE if args[CLI.ARG_GRANULARITY] == 'sentence' else TokenGranularity.WORD

    bench = Benchmark(args[CLI.ARG_SRC], args[CLI.ARG_METHOD], granularity, logger)
    bench.run()