import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ManuscriptGenerationTests(unittest.TestCase):
    def test_markdown_source_exists(self):
        self.assertTrue(Path("docs/manuscript_v4.md").exists())

    def test_docx_generation_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "paper.docx"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_paper_v3.py",
                    "--source",
                    "docs/manuscript_v4.md",
                    "--output",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Generated", result.stdout)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
