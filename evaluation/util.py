
import Levenshtein
from typing import *

def word_error_rate(pred: List[str], target: List[str]) -> float:
    """
	Computes the Word Error Rate, defined as the edit distance between the
	two provided sentences after tokenizing to words.
	:param pred: List of predicted words.
	:param target: List of ground truth words.
	:return:
	"""
    if not isinstance(pred, list) or not isinstance(target, list):
        raise ValueError("word_error_rate: Arguments should be array of words")
    # build mapping of words to integers
    b = set(pred + target)
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts strings)
    w1 = [chr(word2char[w]) for w in pred]
    w2 = [chr(word2char[w]) for w in target]

    d = Levenshtein._levenshtein.distance("".join(w1), "".join(w2))
    wer = d / max(len(target), 1)
    wer = min(wer, 1.0)
    return wer