import re
import spacy

from typing import List

from speechless.edit_context import TimelineChange

TOKEN_SEPARATOR = ' '
SENTENCE_SEPARATOR = '.'
SENTENCE_SEPARATOR_ALTERNATIVES = [',']

SPACY_MODEL = 'en_core_web_md'


class EditToken:
  TEXT_SUB_PATTERNS = re.compile(r'(\[[^\]]*\])|'  # text in []
                                 r'(\([^\)]*\))')  # text in ()
  WHITESPACE_SUB_PATTERNS = re.compile(r'(^\s+)|'  # whitespaces at the beginning
                                       r'(\s+$)|'  # whitespaces at the end
                                       r'((?<=\s)\s+)')  # multiple whitespaces

  def __init__(self, text: str, start_time: float, end_time: float):
    """Some part of the transcript, for which the timestamps are known. The text of the token is
    being normalized, which might reduce it to an empty string - this should be properly handled.
    See `TEXT_SUB_PATTERNS` and `WHITESPACE_SUB_PATTERNS` for patterns, which are removed from the
    text of the token.

    Args:
        text (str): Text that occurred in the transcript of the recording
        start_time (float): The start time of the occurrence
        end_time (float): The end time of the occurrence
    """
    self.text = self.TEXT_SUB_PATTERNS.sub('', text)
    self.text = self.WHITESPACE_SUB_PATTERNS.sub('', self.text)
    self.start_time = start_time
    self.end_time = end_time
    self.start_pos = None  # position in the document (character)
    self.index = None  # index of the token in the document
    self.label = None
    assert self.start_time < self.end_time

  def as_timeline_change(self, duration_ratio: float) -> TimelineChange:
    return TimelineChange(self.start_time, self.end_time, duration_ratio)

  def __len__(self) -> int:
    return len(self.text)


def spacy_nlp(text: str) -> spacy.tokens.Doc:
  """Runs spaCy on a provided text. This function uses a static instance of spaCy language, so it is
  initialized only once.

  Args:
      text (str): Text to process

  Returns:
      spacy.tokens.Doc: spaCy document generated from the text
  """
  if not hasattr(spacy_nlp, 'nlp'):  # lazy initialization of the spaCy Language
    spacy_nlp.nlp = spacy.load(SPACY_MODEL,
                               disable=['tagger', 'attribute_ruler', 'lemmatizer', 'ner'])
  return spacy_nlp.nlp(text)


def sentence_segmentation(transcript: List[EditToken]) -> List[List[EditToken]]:
  """Segments a transcript into sentences

  Args:
      transcript (List[EditToken]): Transcript to segment

  Returns:
      List[List[EditToken]]: Segmented transcript. Each list of tokens is a separate sentence
  """
  # 0. Add a separator between tokens
  transcript[0].start_pos = 0
  for token, next_token in zip(transcript[:-1], transcript[1:]):
    token.text += TOKEN_SEPARATOR
    next_token.start_pos = token.start_pos + len(token)

  # assign indices (useful when working with sentences)
  for idx, token in enumerate(transcript):
    token.index = idx

  raw_transcript = ''.join([token.text for token in transcript])
  sentences: List[List[EditToken]] = []

  # 1. Segment using spaCy
  doc = spacy_nlp(raw_transcript)
  nlp_sents = list(doc.sents)
  assert nlp_sents[0].start_char == 0

  # Assign tokens to the sentences generated by spaCy
  token_idx = 0
  for sent in nlp_sents:
    sentences.append([])
    while token_idx < len(transcript):
      token = transcript[token_idx]
      if token.start_pos < sent.end_char:
        # note, that here we only check if the token starts within the current sentence, and not
        # if it ends inside it aswell. This means, that if a token extends outside of the current
        # sentence, this sentence will consume some part (or all) of the next sentence(s)
        sentences[-1].append(token)
        token_idx += 1
      else:
        break
    if len(sentences[-1]) == 0:
      # when the tokens of this sentence were already consumed by some previous sentence
      sentences.pop()

  # 2. Segment by the time between tokens
  sent_idx = 0
  while sent_idx < len(sentences):
    sent = sentences[sent_idx]
    # sent_len = (sent[-1].start_pos - sent[0].start_pos) + len(sent[-1])
    # if sent_len > 20 * 5 * 2:  # (avg_sent_length) * (avg_eng_word) * 2
    for prev_token_idx, (prev_token, token) in enumerate(zip(sent[:-1], sent[1:])):
      if (token.start_time - prev_token.end_time) > 3.0:
        token_idx = prev_token_idx + 1
        new_sent = sent[token_idx:]
        sentences[sent_idx] = sent[:token_idx]
        sentences.insert(sent_idx + 1, new_sent)
        break
    sent_idx += 1

  # 3. Add a sentence separator between sentences
  pos_diff = 0
  for sent, next_sent in zip(sentences[:-1], sentences[1:]):
    if sent[-1].text[-len(SENTENCE_SEPARATOR):] != SENTENCE_SEPARATOR:
      for alternative in SENTENCE_SEPARATOR_ALTERNATIVES:
        alternative += TOKEN_SEPARATOR  # all tokens had TOKEN_SEPARATOR added at the end
        if sent[-1].text.endswith(alternative):
          sent[-1].text = sent[-1].text[:-len(alternative)] + SENTENCE_SEPARATOR
          pos_diff += len(SENTENCE_SEPARATOR) - len(alternative)
          break

      if sent[-1].text.endswith(SENTENCE_SEPARATOR + TOKEN_SEPARATOR):
        sent[-1].text = sent[-1].text[:-len(TOKEN_SEPARATOR)]
        pos_diff -= len(TOKEN_SEPARATOR)
      elif sent[-1].text.endswith(TOKEN_SEPARATOR):
        sent[-1].text = sent[-1].text[:-len(TOKEN_SEPARATOR)] + SENTENCE_SEPARATOR
        pos_diff += len(SENTENCE_SEPARATOR) - len(TOKEN_SEPARATOR)
      else:
        assert sent[-1].text.endswith(SENTENCE_SEPARATOR)
      for token in next_sent:
        token.start_pos += pos_diff

  return sentences
