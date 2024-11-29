#!/usr/bin/env python3

# Ruff Settings
# ruff: noqa: PT009

import argparse
import inspect
import random
import unittest

import numpy as np
from evaluation_utils import CustomTestCase, Test

from src import main
from utils import read_corpus


def toy_corpus():
    """Generate a toy corpus for testing purposes.

    Returns
    -------
    list of list of str
        A small corpus represented as a list of sentences, each sentence is a list of words.
    """
    return [
        "START All that glitters isn't gold END".split(' '),
        "START All's well that ends well END".split(' '),
    ]


def toy_corpus_co_occurrence():
    """Provide the co-occurrence matrix and word-to-index mapping for the toy corpus.

    Returns
    -------
    m : numpy.ndarray
        Co-occurrence matrix for the toy corpus.
    word2ind : dict
        Mapping from words to their indices in the co-occurrence matrix.
    """
    # Co-occurrence matrix for toy_corpus with window_size = 2
    m = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        ]
    )
    word2ind = {
        'All': 0,
        "All's": 1,
        'END': 2,
        'START': 3,
        'ends': 4,
        'glitters': 5,
        'gold': 6,
        "isn't": 7,
        'that': 8,
        'well': 9,
    }
    return m, word2ind


# > Tests

class Test1(CustomTestCase):
    """Unit tests for the functions defined in main.py."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)

    @Test()
    def test_0(self):
        """Test distinct_words() with a toy corpus."""
        test_corpus = toy_corpus()
        test_corpus_words, num_corpus_words = main.distinct_words(test_corpus)

        expected_words = sorted(
            ['START', 'All', 'ends', 'that', 'gold', "All's", 'glitters', "isn't", 'well', 'END']
        )
        expected_num_words = len(expected_words)

        self.assertEqual(test_corpus_words, expected_words)
        self.assertEqual(num_corpus_words, expected_num_words)

    @Test()
    def test_1(self):
        """Test compute_co_occurrence_matrix() with a toy corpus."""
        test_corpus = toy_corpus()
        m_test, word2ind_test = main.compute_co_occurrence_matrix(test_corpus, window_size=2)

        m_expected, word2ind_expected = toy_corpus_co_occurrence()

        for word1 in word2ind_expected:
            idx1 = word2ind_expected[word1]
            for word2 in word2ind_expected:
                idx2 = word2ind_expected[word2]
                student_value = m_test[idx1, idx2]
                expected_value = m_expected[idx1, idx2]
                self.assertEqual(
                    student_value,
                    expected_value,
                    f'Incorrect count at index ({idx1}, {idx2})=({word1}, {word2}) in matrix m.'
                    f' Yours has {student_value} but should have {expected_value}.',
                )
        self.assertEqual(m_test.shape, m_expected.shape)
        self.assertEqual(word2ind_test, word2ind_expected)

    @Test()
    def test_2(self):
        """Test reduce_to_k_dim() with the toy corpus co-occurrence matrix."""
        m_test, _ = toy_corpus_co_occurrence()
        m_reduced = main.reduce_to_k_dim(m_test, k=2)
        self.assertEqual(m_reduced.shape, (10, 2))

    @Test(timeout=20)
    def test_3(self):
        """Test distinct_words() with the full corpus."""
        corpus = read_corpus()
        student_result, num_words = main.distinct_words(corpus.copy())

        self.assertIsInstance(student_result, list)
        self.assertGreater(len(student_result), 0)
        self.assertIsInstance(num_words, int)
        self.assertEqual(len(student_result), num_words)

    @Test(timeout=20)
    def test_4(self):
        """Test compute_co_occurrence_matrix() with the full corpus."""
        corpus = read_corpus()
        window_size = 4
        student_matrix, student_dict = main.compute_co_occurrence_matrix(
            corpus.copy(), window_size
        )

        self.assertIsInstance(student_matrix, np.ndarray)
        self.assertIsInstance(student_dict, dict)
        self.assertEqual(student_matrix.shape[0], len(student_dict))
        self.assertEqual(student_matrix.shape[1], len(student_dict))

    @Test()
    def test_5(self):
        """Test reduce_to_k_dim() with a random matrix."""
        random.seed(35436)
        np.random.seed(4355)

        x = 10 * np.random.rand(50, 100) + 100
        k = 5

        student_result = main.reduce_to_k_dim(x.copy(), k)
        self.assertEqual(student_result.shape, (50, k))


def get_test_case_for_test_id(test_id):
    """Retrieve the test case corresponding to the given test ID.

    Parameters
    ----------
    test_id : str
        The test ID in the format 'question-part-...'.

    Returns
    -------
    unittest.TestCase
        The test case corresponding to the test ID.
    """
    question, part, _ = test_id.split('-')
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ('Test' + question):
            return obj('test_' + part)
    return None


if __name__ == '__main__':
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument('test_case', nargs='?', default='all')
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != 'all':
        assignment.addTest(get_test_case_for_test_id(test_id))
    else:
        assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='evaluator.py'))
    unittest.TextTestRunner().run(assignment)
