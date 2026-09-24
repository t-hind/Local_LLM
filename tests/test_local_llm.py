import unittest
from pathlib import Path
from unittest.mock import patch

from local_llm import LocalModel, build_prompt


class LocalLlmTests(unittest.TestCase):
    def test_build_prompt_contains_turns(self) -> None:
        prompt = build_prompt("Be brief.", "What is 2 + 2?")
        self.assertEqual(
            prompt,
            "<|system|>\nBe brief.\n<|user|>\nWhat is 2 + 2?\n<|assistant|>\n",
        )

    def test_empty_prompt_is_rejected_before_loading(self) -> None:
        model = LocalModel(Path("missing.gguf"), 1024, 1)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            model.generate(" ")

    def test_health_model_stays_lazy(self) -> None:
        model = LocalModel(Path("missing.gguf"), 1024, 1)
        self.assertIsNone(model._model)


if __name__ == "__main__":
    unittest.main()