"""Part 2 (Gemini variant) -- measurement using the Google Gemini API.

Does the same job as ``part2_measure.py``, but built for a Google AI Studio
key instead of an Anthropic key. Anthropic and Gemini are two different
APIs with two different tokenizers and two different price lists, so the
numbers this produces describe *Gemini*, not Claude -- Part 3 will report
Gemini's cost structure once it reads the file this script writes.

Does two things:

1. One real request per language, so you can see a response and the token
   usage that comes back with it. Three calls, because the answer length is
   itself language-dependent -- assuming it away is how you get a wrong bill.
   This is the only part of the lab that spends money (Gemini has a free
   tier for these small, low-volume flash calls, so this is normally $0).
2. Token counts for every text in :mod:`texts`, in every language, using the
   ``count_tokens`` endpoint. Counting tokens is free and does not run the
   model. The combined "one real request" count (system prompt + complaint)
   comes from --call's own billed usage when available (see below).

Needs the ``google-genai`` package and a Gemini API key:

    pip install google-genai
    # get a key at https://aistudio.google.com/apikey
    # put it in .env as: GEMINI_API_KEY=your-key-here

Run:
    python3 part2_measure_gemini.py --call              # recommended: exact numbers
    python3 part2_measure_gemini.py                      # count tokens only (estimate for "request")
    python3 part2_measure_gemini.py --call --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

from texts import CORPUS, LANGUAGES

#: Loads GEMINI_API_KEY from a .env file next to this script, if present.
#: A real environment variable, if already set, is left untouched.
load_dotenv()

OUTPUT_PATH = Path(__file__).with_name("measurements.json")

#: Gemini model ids this script knows how to price. Keep in sync with the
#: GEMINI_MODELS table in prices.py -- these are the keys Part 3 looks up.
#: gemini-2.5-flash was retired for new API keys in September 2026 -- Google's
#: own 404 response now tells new accounts to use gemini-3.6-flash instead.
#: Both are kept here: an older key may still resolve 2.5-flash, a new key
#: will not.
GEMINI_MODELS = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3.6-flash": "gemini-3.6-flash",
}
DEFAULT_MODEL = "gemini-3.6-flash"

#: Same reasoning as part2_measure.py's MAX_TOKENS: bounded, not tiny.
#: Gemini's thinking models also spend invisible tokens before the visible
#: answer -- keep headroom so a real answer isn't cut off mid-word.
MAX_OUTPUT_TOKENS = 2048


def build_client():
    """Return an authenticated genai.Client, or exit with a usable message."""
    try:
        from google import genai
    except ImportError:
        sys.exit(
            "google-genai is not installed.\n\n"
            "  pip install google-genai\n"
        )
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit(
            "no API key found. Get one at https://aistudio.google.com/apikey, "
            "then either:\n"
            "  - put GEMINI_API_KEY=your-key in a .env file next to this script, or\n"
            "  - set it in the shell: export GEMINI_API_KEY=your-key "
            "(Windows: set GEMINI_API_KEY=your-key)"
        )
    return genai.Client(api_key=api_key)


def count_tokens(client, model_id: str, text: str) -> int:
    """Return the number of tokens ``text`` costs on ``model_id``.

    Token counts are model-specific: Gemini's tokenizer differs from
    Claude's and from tiktoken's, so this is a third, separate measurement,
    not interchangeable with Part 0 or Part 2's Claude version.
    """
    result = client.models.count_tokens(model=model_id, contents=text)
    return result.total_tokens


def count_request_tokens_estimate(client, model_id: str, lang: str) -> int:
    """Approximate input tokens for one real request, when --call was not used.

    As of late 2026 the Gemini Developer API's count_tokens endpoint rejects
    a system_instruction in its config ("system_instruction parameter is
    only supported in Gemini Enterprise Agent Platform mode, not in Gemini
    Developer API mode") -- so the exact one-call shape Part 3 wants cannot
    be counted for free anymore. This concatenates system prompt and
    complaint into one block of text and counts that instead. It is an
    ESTIMATE: it will not match the true billed prompt_token_count exactly,
    because it does not reproduce whatever separator/role framing the API
    inserts between a system instruction and a user message internally.
    Prefer running with --call, which reports the true count from a real
    response's usage_metadata.
    """
    combined = CORPUS["system_prompt"][lang] + "\n\n" + CORPUS["complaint"][lang]
    result = client.models.count_tokens(model=model_id, contents=combined)
    return result.total_tokens


#: Free-tier Gemini calls occasionally hit a transient 503 ("high demand" /
#: UNAVAILABLE) that has nothing to do with this request -- retrying after a
#: short wait usually succeeds. This bounds how many times, and how long,
#: one_real_request will wait before giving up and re-raising.
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5


def one_real_request(client, model_id: str, lang: str) -> Optional[Dict[str, int]]:
    """Send one request and print the answer and its billed usage.

    Retries a few times on a transient server error (503 UNAVAILABLE, "high
    demand") with a short backoff, since that failure is about Google's
    load, not this request, and usually clears within seconds.
    """
    from google.genai import errors, types

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=CORPUS["complaint"][lang],
                config=types.GenerateContentConfig(
                    system_instruction=CORPUS["system_prompt"][lang],
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
            break
        except errors.ServerError as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"  server error ({exc}), retrying in {wait}s "
                  f"[{attempt}/{MAX_RETRIES}]...")
            time.sleep(wait)

    usage = response.usage_metadata
    if usage is None:
        print("  no usage_metadata returned -- the call may have been blocked")
        return None

    text = getattr(response, "text", None)
    print("  --- answer ---")
    print("  " + (text or "(empty -- check response.candidates[0].finish_reason)")
          .replace("\n", "\n  "))

    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    # Gemini 2.5's "thinking" models spend tokens the visible answer never
    # shows, billed at the output rate -- fold them in, same trap as opus-5
    # in part2_measure.py.
    thinking_tokens = getattr(usage, "thoughts_token_count", None) or 0
    total_output = output_tokens + thinking_tokens

    print(f"  billed: {input_tokens} in, {output_tokens} out"
          + (f" + {thinking_tokens} thinking" if thinking_tokens else ""))
    return {"input_tokens": input_tokens, "output_tokens": total_output}


def main() -> int:
    """Count tokens for the whole corpus, optionally make one real request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, choices=sorted(GEMINI_MODELS),
        help=f"which Gemini model to price against (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--call", action="store_true",
        help="also answer the complaint in each language (usually free on flash)",
    )
    args = parser.parse_args()

    model_id = GEMINI_MODELS[args.model]
    client = build_client()

    counts: Dict[str, Dict[str, int]] = {}
    print(f"counting tokens on {model_id} (free, no model run)")
    for item_id, versions in CORPUS.items():
        counts[item_id] = {
            lang: count_tokens(client, model_id, versions[lang])
            for lang in LANGUAGES
        }
        row = "  ".join(f"{lang}={counts[item_id][lang]}" for lang in LANGUAGES)
        print(f"  {item_id:<14} {row}")

    billed: Dict[str, Dict[str, int]] = {}
    if args.call:
        print(f"\nanswering the same complaint on {model_id}, in each language:")
        for lang in LANGUAGES:
            print(f"\n[{lang}]")
            result = one_real_request(client, model_id, lang)
            if result is not None:
                billed[lang] = result

    if billed and all(lang in billed for lang in LANGUAGES):
        # Exact: the real request's own billed usage already reports system
        # prompt + complaint, one call -- no separate count_tokens call
        # needed, and none of count_tokens' API restrictions apply here.
        request_tokens: Dict[str, int] = {
            lang: billed[lang]["input_tokens"] for lang in LANGUAGES
        }
        provenance = "measured from the real --call request"
    else:
        # Fallback when running without --call: the Developer API's
        # count_tokens no longer accepts a system_instruction (see
        # count_request_tokens_estimate's docstring), so this is an
        # approximation, not the exact billed number.
        request_tokens = {
            lang: count_request_tokens_estimate(client, model_id, lang)
            for lang in LANGUAGES
        }
        provenance = "ESTIMATED (concatenated system+complaint text; run with --call for the exact number)"
    row = "  ".join(f"{lang}={request_tokens[lang]}" for lang in LANGUAGES)
    print(f"\n  {'request':<14} {row}  ({provenance})")

    payload = {
        "provider": "gemini",
        "model": args.model,
        "model_id": model_id,
        "token_counts": counts,
        "request_tokens": request_tokens,
        "one_request_billed": billed or None,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUTPUT_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())