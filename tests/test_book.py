import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from helpers import make_epub  # noqa: E402
from soundscapes.clean import normalize  # noqa: E402
from soundscapes.book import process_epub, write_outputs  # noqa: E402
from soundscapes.segment import MIN_CHARS, sentence_spans  # noqa: E402


class SentenceSplit(unittest.TestCase):
    def split(self, t):
        return [t[s:e] for s, e in sentence_spans(t)]

    def test_basic(self):
        self.assertEqual(self.split('He shut it. It was dark! Was it? Yes.'),
                         ['He shut it.', 'It was dark!', 'Was it?', 'Yes.'])

    def test_dialogue_tag_not_split(self):
        self.assertEqual(self.split('"Good morning!" said Bilbo. He meant it.'),
                         ['"Good morning!" said Bilbo.', 'He meant it.'])

    def test_closing_quote_kept(self):
        self.assertEqual(self.split('"Go away." Then he left.'), ['"Go away."', 'Then he left.'])

    def test_abbreviation(self):
        self.assertEqual(self.split('Mr. Peter heard it. He went home.'), ['Mr. Peter heard it.', 'He went home.'])

    def test_normalize(self):
        self.assertEqual(normalize('\u201cHi,\u201d  she\u2019s\nhere\u2014now'), '"Hi," she\'s here - now')


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        epub = make_epub(os.path.join(cls.tmp.name, 'goblin_mini.epub'), ['goblin_mini.xhtml'])
        cls.text, cls.segs, cls.chapters = process_epub(epub)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_front_matter_and_toc_dropped(self):
        self.assertNotIn('GOBLIN', self.text)
        self.assertNotIn('Table of Contents', self.text)
        self.assertEqual([(c['chapter'], c['title']) for c in self.chapters],
                         [(1, 'Why the Princess Has a Story About Her'), (2, 'The Princess Loses Herself')])

    def test_offsets_and_ids(self):
        for i, s in enumerate(self.segs):
            self.assertEqual(s['id'], i)
            self.assertEqual(self.text[s['char_start']:s['char_end']], s['text'])

    def test_min_length_merge(self):
        texts = [s['text'] for s in self.segs]
        self.assertTrue(all(len(t) >= MIN_CHARS for t in texts))
        # A short segment absorbs the next one, across paragraphs ...
        self.assertIn('"Why?" asked Irene.\n\n"Because it is late."', texts)
        # ... and at a chapter end it joins the one before.
        self.assertEqual(texts[-1], 'Mr. Peter heard it from far down in the mine. He went home.')

    def test_broken_paragraph_rejoined(self):
        self.assertTrue(any('would have thought must have known' in s['text'] for s in self.segs))

    def test_verse_detected(self):
        verse = [s['text'] for s in self.segs if s['is_verse']]
        self.assertEqual(verse[0], "'Ring! dod! bang!\nGo the hammers' clang!")
        self.assertFalse(any(s['is_verse'] for s in self.segs if 'sang as they worked' in s['text']))
        self.assertFalse(any(s['is_verse'] for s in self.segs if s['chapter'] == 1))

    def test_outputs_match_frontend_format(self):
        files = write_outputs(self.tmp.name, 'gob', 'Goblin', self.text, self.segs, self.chapters)
        doc = json.load(open(files[2], encoding='utf-8'))
        self.assertEqual(doc['book'], 'Goblin')
        self.assertEqual(open(files[0], encoding='utf-8').read(), self.text)
        self.assertEqual(list(doc['segments'][0]), ['id', 'char_start', 'char_end', 'text', 'has_sound', 'clip'])


if __name__ == '__main__':
    unittest.main()
