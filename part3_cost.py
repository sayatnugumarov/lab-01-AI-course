"""Part 3 -- turn the measured token counts into money.

Reads ``measurements.json`` from Part 2 and answers the three questions the
lab exists for:

* what one support request costs, in each language;
* what a year of them costs at a realistic volume;
* how much more a Kazakh-language service costs than an English one, for
  exactly the same work.

Watch the last table. The input-token ratio and the total-bill ratio are
different numbers, and quoting one while meaning the other is the most common
way to be wrong about this in public.

Run:
    python3 part3_cost.py
    python3 part3_cost.py --requests-per-day 5000
    python3 part3_cost.py --output-tokens 300     # override the measurement
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from prices import (CLAUDE_MODEL_ORDER, GEMINI_MODEL_ORDER, GEMINI_PRICE_CHECKED,
                    GEMINI_PRICE_SOURCE, MODELS, PRICE_CHECKED, PRICE_SOURCE,
                    cost_usd)
from texts import LANGUAGES

DEFAULT_MEASUREMENTS = Path(__file__).with_name("measurements.json")

#: Used only when Part 2 ran without --call, so no answer was ever generated.
FALLBACK_OUTPUT_TOKENS = 300

#: measurements.json's "provider" field (written by part2_measure.py as
#: "anthropic" implicitly, or by part2_measure_gemini.py as "gemini")
#: decides which price table and model order apply -- the two providers'
#: model ids are disjoint keys in the same MODELS dict, but printing the
#: wrong provider's table would silently price Gemini tokens at Claude
#: rates or vice versa.
PROVIDER_SETTINGS = {
    "anthropic": (CLAUDE_MODEL_ORDER, PRICE_SOURCE, PRICE_CHECKED, "opus-5"),
    "gemini": (GEMINI_MODEL_ORDER, GEMINI_PRICE_SOURCE, GEMINI_PRICE_CHECKED,
               "gemini-2.5-flash"),
}


def load_measurements(path: Path) -> Dict[str, object]:
    """Read the JSON written by Part 2.

    Args:
        path: Path to ``measurements.json``.

    Returns:
        The parsed payload.

    Raises:
        SystemExit: If the file is missing or is not valid JSON. There is no
            fallback on purpose -- this lab runs on measured numbers only.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(
            f"{path.name} not found. Run part2_measure.py first -- "
            "this script will not invent token counts."
        )
    except json.JSONDecodeError as exc:
        sys.exit(f"{path.name} is not valid JSON: {exc}")


def request_input_tokens(data: Dict[str, object], lang: str) -> int:
    """Tokens billed for one real request: system prompt + complaint, combined.

    Reads the value Part 2 measured as a single ``system=`` + ``messages=``
    call -- the exact shape :func:`part2_measure.one_real_request` sends and
    bills. Summing the system prompt's and the complaint's *standalone*
    token counts would double-count each message's per-call framing
    overhead (on this corpus, about 5 tokens out of 145 -- roughly 3%, and
    proportionally larger on the short ``sentence`` item).

    Args:
        data: The full payload written by Part 2.
        lang: Language code, one of :data:`texts.LANGUAGES`.

    Returns:
        Total input tokens for one request in that language.

    Raises:
        SystemExit: If ``data`` predates the ``request_tokens`` field --
            re-run Part 2 rather than falling back to the double-counting
            approximation.
    """
    request_tokens = data.get("request_tokens")
    if not request_tokens or lang not in request_tokens:
        sys.exit(
            "measurements.json has no request_tokens for "
            f"{lang!r} -- re-run part2_measure.py to regenerate it."
        )
    return request_tokens[lang]


def resolve_output_tokens(
    billed: Optional[Dict[str, Dict[str, int]]], override: Optional[int]
) -> Tuple[Dict[str, int], str]:
    """Decide how many output tokens to charge for, per language.

    Args:
        billed: The ``one_request_billed`` mapping from Part 2, or ``None``.
        override: A fixed answer length supplied on the command line.

    Returns:
        A ``(tokens_by_language, provenance)`` pair, where the second element
        says in plain words where the numbers came from.
    """
    if override is not None:
        return {lang: override for lang in LANGUAGES}, "fixed by --output-tokens"
    if billed and all(lang in billed for lang in LANGUAGES):
        return (
            {lang: billed[lang]["output_tokens"] for lang in LANGUAGES},
            "measured in Part 2, per language",
        )
    return (
        {lang: FALLBACK_OUTPUT_TOKENS for lang in LANGUAGES},
        "ASSUMED -- Part 2 ran without --call, so no answer was measured",
    )


def _header() -> str:
    """Column header shared by every table."""
    return f"{'':<14}" + "".join(f"{lang.upper():>12}" for lang in LANGUAGES)


def main() -> int:
    """Print the per-request, annual and cross-language cost tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements", type=Path, default=DEFAULT_MEASUREMENTS,
        help="path to the JSON written by part2_measure.py",
    )
    parser.add_argument(
        "--requests-per-day", type=int, default=2000,
        help="support volume to project (default: 2000)",
    )
    parser.add_argument(
        "--output-tokens", type=int, default=None,
        help="force one answer length for all languages, instead of the measurement",
    )
    parser.add_argument(
        "--model", default=None, choices=sorted(MODELS),
        help="which model the final summary is about "
             "(default: opus-5 for Claude data, gemini-2.5-flash for Gemini data)",
    )
    args = parser.parse_args()

    data = load_measurements(args.measurements)
    provider = data.get("provider", "anthropic")
    if provider not in PROVIDER_SETTINGS:
        sys.exit(f"unknown provider {provider!r} in {args.measurements.name}")
    MODEL_ORDER, price_source, price_checked, default_model = PROVIDER_SETTINGS[provider]
    model = args.model or default_model
    if model not in MODEL_ORDER:
        sys.exit(
            f"--model {model} is not a {provider} model -- "
            f"choose one of: {', '.join(MODEL_ORDER)}"
        )
    args.model = model

    outputs, provenance = resolve_output_tokens(data.get("one_request_billed"),
                                                args.output_tokens)

    print(f"provider: {provider}")
    print(f"prices from {price_source}")
    print(f"checked {price_checked}; tokens counted on {data['model_id']}")
    print(f"answer length: {provenance}\n")

    inputs = {lang: request_input_tokens(data, lang) for lang in LANGUAGES}

    print("ONE SUPPORT REQUEST -- tokens, and cost in US cents")
    print("-" * 72)
    print(_header())
    print(f"{'input tokens':<14}" + "".join(f"{inputs[l]:>12}" for l in LANGUAGES))
    print(f"{'output tokens':<14}" + "".join(f"{outputs[l]:>12}" for l in LANGUAGES))
    for model_key in MODEL_ORDER:
        cents = [
            cost_usd(model_key, inputs[lang], outputs[lang]) * 100
            for lang in LANGUAGES
        ]
        print(f"{model_key:<14}" + "".join(f"{c:>12.4f}" for c in cents))

    per_year = args.requests_per_day * 365
    print(f"\nAT {args.requests_per_day:,} REQUESTS/DAY -- US dollars per year")
    print("-" * 72)
    print(_header())
    for model_key in MODEL_ORDER:
        yearly = [
            cost_usd(model_key, inputs[lang], outputs[lang]) * per_year
            for lang in LANGUAGES
        ]
        print(f"{model_key:<14}" + "".join(f"{y:>12,.0f}" for y in yearly))

    print("\nTWO RATIOS THAT ARE NOT THE SAME NUMBER")
    print("-" * 72)
    print(_header())
    print(
        f"{'input only':<14}"
        + "".join(f"{inputs[l] / inputs['en']:>11.2f}x" for l in LANGUAGES)
    )
    base_bill = cost_usd(args.model, inputs["en"], outputs["en"])
    print(
        f"{'total bill':<14}"
        + "".join(
            f"{cost_usd(args.model, inputs[l], outputs[l]) / base_bill:>11.2f}x"
            for l in LANGUAGES
        )
    )
    print(
        "\nThe first row is a property of the tokenizer. The second is what you\n"
        f"actually pay on {args.model}, and it moves with the length of the answer.\n"
        "State which one you mean, every time, or the number is worthless."
    )

    print("\nTHE NUMBER TO REMEMBER")
    print("-" * 72)
    en_year = cost_usd(args.model, inputs["en"], outputs["en"]) * per_year
    for lang in ("ru", "kk"):
        lang_year = cost_usd(args.model, inputs[lang], outputs[lang]) * per_year
        print(
            f"{lang.upper()} instead of EN on {args.model}, same work, same volume: "
            f"${lang_year - en_year:,.0f}/year more ({lang_year / en_year:.2f}x)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())