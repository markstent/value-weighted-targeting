import unittest
from pathlib import Path


class ManuscriptGenerationTests(unittest.TestCase):
    def test_manuscript_source_exists(self):
        self.assertTrue(Path("docs/manuscript_v13.tex").exists())

    def test_manuscript_pdf_exists(self):
        self.assertTrue(Path("docs/manuscript_v13.pdf").exists())


if __name__ == "__main__":
    unittest.main()
