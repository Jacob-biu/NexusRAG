from hashlib import md5
from dataclasses import dataclass, field
from typing import List, Dict
import httpx
from openai import OpenAI
from collections import defaultdict
import multiprocessing as mp
import re
import string
import logging
import numpy as np
import os
import time
import argparse
from httpx import Limits
from dotenv import load_dotenv

# Load the .env file in the same directory
load_dotenv()

# Default local vLLM configuration
DEFAULT_BASE_URL = "http://localhost:8009/v1"


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


class LLM_Model:
    def __init__(self, llm_model, base_url: str = None):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        # Priority: explicit argument > environment variable > default URL
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

        self.timeout = 600

        # Single place to configure every LLM parameter
        self.llm_config = {
            "model": llm_model,
            "max_tokens": 10000,
            "temperature": 0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "seed": 42,
        }

        if (
            llm_model == "Qwen3.6-27B-FP8"
        ):
            self.llm_config["chat_template_kwargs"] = {"enable_thinking": False}

        # Request headers: add the auth header only when an API key is present
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # HTTP client configuration
        self.http_client = httpx.Client(
            timeout=self.timeout,
            trust_env=False,  # Disable system proxies so local requests are not intercepted
            verify=False,  # No SSL verification needed for a local HTTP service
            headers=headers,
            limits=Limits(max_connections=100, max_keepalive_connections=20),
        )

        # Suppress insecure-request warnings
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def infer(self, messages):
        """
        LLM inference interface, using the shared llm_config parameters.
        Never raises on timeout or error: it always returns content (an error message on failure).
        """
        url = f"{self.base_url}/chat/completions"
        payload = {**self.llm_config, "messages": messages}

        max_retry = 3
        retry_count = 0
        while retry_count < max_retry:
            try:
                response = self.http_client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                retry_count += 1
                error_msg = f"LLM request failed (retry {retry_count}/{max_retry}): {str(e)}"
                if retry_count >= max_retry:
                    logging.error(error_msg)
                if retry_count >= max_retry:
                    return error_msg

    def test_connection(self) -> bool:
        """Connection test helper, using the same shared llm_config."""
        print(f"Testing the connection to: {self.base_url}")
        print(f"Model: {self.llm_config['model']}")
        print("-" * 50)

        try:
            url = f"{self.base_url}/chat/completions"

            # Test requests may override values from llm_config
            payload = {
                **self.llm_config,
                "messages": [
                    {"role": "user", "content": "Reply with the single word 'ok'."}
                ],
                "max_tokens": 10,  # Override to limit the token usage
            }

            response = self.http_client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            result_content = result["choices"][0]["message"]["content"].strip()
            print(f"✅ Connection succeeded. Model reply: {result_content}")
            print(f"🔍 Tokens used: {result['usage']['total_tokens']}")
            return True

        except Exception as e:
            print(f"❌ Connection failed. Error type: {type(e).__name__}")
            print(f"❌ Error message: {str(e)}")
            print("-" * 50)
            print("🔧 Troubleshooting:")
            print("1. Check that the vLLM service is running")
            print("2. Check that base_url is correct")
            print("3. Check that the model name matches the deployed model exactly")
            print("4. Check that the port is open and free of conflicts")
            print("5. Check that vLLM finished loading the model, with no errors in its startup log")
            print("6. Try upgrading httpx: pip install -U httpx")
            return False


def normalize_answer(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def setup_logging(log_file):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO, format=log_format, handlers=handlers, force=True
    )
    # Suppress noisy HTTP request logs (e.g., 401 Unauthorized) from httpx/openai
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def min_max_normalize(x):
    # Clamp negative values to 0
    x = np.maximum(x, 0.0)

    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val

    # Return an all-ones array when all values are identical
    if range_val == 0:
        return np.ones_like(x)

    return (x - min_val) / range_val


if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Local vLLM model client")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="local vLLM service URL; defaults to http://localhost:8009/v1 when omitted",
    )
    args = parser.parse_args()

    # IMPORTANT: set this to the model name actually deployed in your local vLLM service
    TEST_MODEL = "Qwen3.6-27B-FP8"

    # Initialize the model with a custom base_url
    llm = LLM_Model(TEST_MODEL, base_url=args.base_url)
    connection_success = llm.test_connection()

    if connection_success:
        print("\n🎉 Connection test passed.")
        test_messages = [{"role": "user", "content": "What is 1+1?"}]
        result = llm.infer(test_messages)
        print(f"\nTest inference result: {result}")
    else:
        print("\n⚠️ Connection test failed, please fix the issue and retry")
