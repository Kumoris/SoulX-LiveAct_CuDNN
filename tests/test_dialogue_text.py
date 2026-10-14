"""Streaming speech boundaries must preserve response text and emit prefixes."""
import unittest
from dialogue.local_backends import sentences


class SentenceTests(unittest.TestCase):
    def test_first_sentence_does_not_wait_for_later_model_tokens(self):
        requested = []
        def tokens():
            requested.append(1)
            yield '你好。'
            requested.append(2)
            yield '我是数字人。'
        phrases = sentences(tokens())
        self.assertEqual(next(phrases), '你好。')
        self.assertEqual(requested, [1])
        self.assertEqual(list(phrases), ['我是数字人。'])

    def test_unpunctuated_stream_has_bounded_prefixes_and_exact_content(self):
        source = '这是一段没有标点但不能丢掉文字的模型回复'*5
        phrases = list(sentences((source[i:i+3] for i in range(0,len(source),3)), max_chars=16))
        self.assertEqual(''.join(phrases), source)
        self.assertTrue(all(len(phrase)<=16 for phrase in phrases))

    def test_chunking_handles_multiple_sentence_endings_in_one_token(self):
        self.assertEqual(list(sentences(['你', '好！明天', '见。尾句'])), ['你好！','明天见。','尾句'])


if __name__ == '__main__':
    unittest.main()
