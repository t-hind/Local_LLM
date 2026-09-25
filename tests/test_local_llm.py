import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from chat_client import Conversation
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

    def test_conversation_includes_previous_turns(self) -> None:
        conversation = Conversation("Be concise.")
        conversation.add_turn("What is 2 + 2?", "4")
        prompt = conversation.prompt_for("What did I ask first?")
        self.assertIn("User: What is 2 + 2?", prompt)
        self.assertIn("Assistant: 4", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_generate_uses_chat_completion_response(self) -> None:
        model = LocalModel(Path("model.gguf"), 1024, 1)
        fake_engine = Mock()
        fake_engine.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "A clean answer."}}]
        }
        model._model = fake_engine
        with patch.object(Path, "is_file", return_value=True):
            answer = model.generate("Hello")
        self.assertEqual(answer, "A clean answer.")
        fake_engine.create_chat_completion.assert_called_once()


if __name__ == "__main__":
    unittest.main()