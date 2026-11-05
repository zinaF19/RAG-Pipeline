"""Provider selection and response guards, without API calls."""

import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from pydantic import SecretStr

import llm_config


class ProviderTests(unittest.TestCase):
    def test_each_provider_uses_its_key_model_and_native_adapter(self):
        for name, (adapter, default) in llm_config.PROVIDERS.items():
            with (
                self.subTest(provider=name),
                patch.dict(
                    os.environ,
                    {"LLM_PROVIDER": name, f"{name.upper()}_API_KEY": "test-only"},
                    clear=True,
                ),
                patch.object(llm_config, "init_chat_model") as initialize,
            ):
                llm_config.create_chat_model()
                options = initialize.call_args.kwargs
                self.assertEqual(options["model_provider"], adapter)
                self.assertEqual(options["model"], default)
                self.assertIsInstance(options["api_key"], SecretStr)
                self.assertEqual(options["api_key"].get_secret_value(), "test-only")
                self.assertEqual(options["max_retries"], 0)
                self.assertEqual(options["max_tokens"], 1024)
                self.assertEqual(
                    options["timeout"], 30000 if name == "openrouter" else 30
                )

    def test_missing_selected_key_does_not_use_another_provider(self):
        with (
            patch.dict(
                os.environ,
                {"LLM_PROVIDER": "mistral", "GROQ_API_KEY": "other"},
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "MISTRAL_API_KEY"),
        ):
            llm_config.create_chat_model()

    def test_unknown_provider_fails_before_any_request(self):
        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "typo"}, clear=True),
            self.assertRaisesRegex(ValueError, "LLM_PROVIDER"),
        ):
            llm_config.create_chat_model()

    def test_overrides_and_openrouter_backup(self):
        with (
            patch.dict(
                os.environ,
                {
                    "LLM_PROVIDER": "openrouter",
                    "OPENROUTER_API_KEY": "test-only",
                    "OPENROUTER_MODEL": "custom/model",
                    "OPENROUTER_FALLBACK_MODEL": "",
                },
                clear=True,
            ),
            patch.object(llm_config, "init_chat_model") as initialize,
        ):
            llm_config.create_chat_model(temperature=0.3, max_tokens=256)
            options = initialize.call_args.kwargs
            self.assertEqual(options["model"], "custom/model")
            self.assertEqual(options["temperature"], 0.3)
            self.assertEqual(options["max_tokens"], 256)
            self.assertIsNone(options["model_kwargs"]["models"])
            self.assertIsNone(options["model_kwargs"]["retries"])

    def test_visible_text_excludes_reasoning(self):
        message = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "Not an answer"},
                {"type": "text", "text": "Answer"},
            ],
            response_metadata={"finish_reason": "STOP"},
        )
        self.assertEqual(llm_config.response_text(message), "Answer")

    def test_incomplete_or_empty_answers_are_rejected(self):
        for text, reason in [
            ("partial", "length"),
            ("", "stop"),
            (" ", "STOP"),
            ("blocked", "SAFETY"),
        ]:
            with self.subTest(reason=reason, text=text), self.assertRaises(ValueError):
                llm_config.response_text(
                    AIMessage(content=text, response_metadata={"finish_reason": reason})
                )


if __name__ == "__main__":
    unittest.main()
