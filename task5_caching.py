"""Extension task 5 -- apply the cost lever the lab defines but never uses.

``prices.py`` ships ``CACHE_READ_FRACTION`` and ``BATCH_DISCOUNT`` and
``cost_usd`` applies neither. This script re-prices the annual bill with
prompt caching switched on for the part of every request that never changes:
the system prompt.

The model
---------
One request bills ``request_tokens[lang]`` input tokens. That total splits in
two, and only one half repeats:

* the system prompt -- identical on every call, so cacheable. Its size is
  ``token_counts.system_prompt[lang]``.
* everything else -- the customer's own complaint plus the per-call message
  framing. Taken as the remainder, ``request_tokens - system_prompt``, so the
  framing is counted exactly once and never double-counted (the trap Part 3's
  docstring and task 4 both warn about).

With caching on, the system-prompt half is billed at
``CACHE_READ_FRACTION[model]`` of the input price on every call after the
first; the remainder stays at full price. Output is untouched -- caching is an
input-side lever only.

Two honest caveats, printed with the numbers:

* This ignores the cache **write**. A real cache write costs more than a
  normal input token (1.25x base on Claude's current pricing), paid once per
  entry and again on every refresh after the TTL lapses. ``prices.py`` has no
  constant for it, so this script does not invent one -- it prints the
  cache-write break-even as a share of daily volume instead.
* ``BATCH_DISCOUNT`` (0.50) is deliberately NOT applied. The Batch API
  processes asynchronously with a completion window measured in hours. A
  customer waiting in a live support queue cannot be served from it, so
  halving this particular bill with it would be a fabricated saving. It
  applies to offline work -- nightly ticket classification, bulk
  summarisation, eval runs -- not to this queue.

Run:
    python3 task5_caching.py
    python3 task5_caching.py --requests-per-day 5000 --model sonnet-5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from prices import (BATCH_DISCOUNT, CACHE_READ_FRACTION, CLAUDE_MODEL_ORDER,
                    GEMINI_CACHE_STORAGE_PER_MTOK_HOUR, GEMINI_MODEL_ORDER,
                    GEMINI_PRICE_CHECKED, GEMINI_PRICE_SOURCE, MODELS,
                    PRICE_CHECKED, PRICE_SOURCE, cost_usd)
from texts import LANGUAGES

DEFAULT_MEASUREMENTS = Path(__file__).with_name("measurements.json")

#: A cache write is billed above the base input rate, once per entry. Claude's
#: published multiplier for a 5-minute TTL; not in prices.py, so it is named
#: here rather than silently folded into the totals. Applies to Claude only --
#: Gemini has no equivalent per-write charge (see PROVIDER_SETTINGS below).
CACHE_WRITE_MULTIPLIER = 1.25

#: Same shape as part3_cost.py's table: which model order, price source and
#: default model apply, keyed by measurements.json's "provider" field.
PROVIDER_SETTINGS = {
    "anthropic": (CLAUDE_MODEL_ORDER, PRICE_SOURCE, PRICE_CHECKED, "opus-5"),
    "gemini": (GEMINI_MODEL_ORDER, GEMINI_PRICE_SOURCE, GEMINI_PRICE_CHECKED,
               "gemini-2.5-flash"),
}


def load(path: Path) -> Dict[str, object]:
    """Read the measurements written by Part 2.

    Args:
        path: Path to ``measurements.json``.

    Returns:
        The parsed payload.

    Raises:
        SystemExit: If the file is missing, malformed, or predates the
            ``request_tokens`` field. No fallback: this lab prices measured
            numbers only.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"{path.name} not found. Run part2_measure.py, or "
                 "cp measurements.example.json measurements.json")
    except json.JSONDecodeError as exc:
        sys.exit(f"{path.name} is not valid JSON: {exc}")
    if "request_tokens" not in data or "token_counts" not in data:
        sys.exit(f"{path.name} is missing request_tokens or token_counts -- "
                 "re-run part2_measure.py.")
    return data


def cached_request_usd(
    model_key: str,
    system_tokens: int,
    variable_tokens: int,
    output_tokens: int,
) -> float:
    """Cost of one request whose system prompt is served from the cache.

    Args:
        model_key: A key of :data:`prices.MODELS`.
        system_tokens: Input tokens in the cached (repeated) system prompt.
        variable_tokens: The rest of the input -- the complaint plus framing.
        output_tokens: Tokens generated.

    Returns:
        Cost in US dollars for one cache-hit request.
    """
    model = MODELS[model_key]
    fraction = CACHE_READ_FRACTION[model_key]
    billed_input = variable_tokens + system_tokens * fraction
    return (billed_input * model.input_per_mtok
            + output_tokens * model.output_per_mtok) / 1_000_000


def main() -> int:
    """Print the uncached and cached annual bills side by side."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--requests-per-day", type=int, default=2000)
    parser.add_argument(
        "--model", default=None, choices=sorted(MODELS),
        help="default: opus-5 for Claude data, gemini-2.5-flash for Gemini data",
    )
    args = parser.parse_args()

    data = load(args.measurements)
    provider = data.get("provider", "anthropic")
    if provider not in PROVIDER_SETTINGS:
        sys.exit(f"unknown provider {provider!r} in {args.measurements.name}")
    model_order, price_source, price_checked, default_model = PROVIDER_SETTINGS[provider]
    model_key = args.model or default_model
    if model_key not in model_order:
        sys.exit(f"--model {model_key} is not a {provider} model -- "
                 f"choose one of: {', '.join(model_order)}")
    per_year = args.requests_per_day * 365

    request_tokens: Dict[str, int] = data["request_tokens"]
    system_tokens: Dict[str, int] = data["token_counts"]["system_prompt"]
    billed = data.get("one_request_billed")
    if not billed or not all(lang in billed for lang in LANGUAGES):
        sys.exit("measurements.json has no per-language output_tokens -- "
                 "re-run part2_measure.py --call.")
    outputs = {lang: billed[lang]["output_tokens"] for lang in LANGUAGES}

    fraction = CACHE_READ_FRACTION[model_key]
    print(f"provider: {provider}")
    print(f"prices from {price_source}")
    print(f"checked {price_checked}; tokens counted on {data['model_id']}")
    print(f"model {model_key}; cache read = {fraction:.1%} of the input price")
    print(f"volume {args.requests_per_day:,}/day = {per_year:,} requests/year\n")

    print("WHAT REPEATS, AND WHAT DOES NOT -- input tokens per request")
    print("-" * 72)
    print(f"{'':<20}" + "".join(f"{lang.upper():>12}" for lang in LANGUAGES))
    variable = {lang: request_tokens[lang] - system_tokens[lang] for lang in LANGUAGES}
    print(f"{'request total':<20}" + "".join(f"{request_tokens[l]:>12}" for l in LANGUAGES))
    print(f"{'system (cacheable)':<20}" + "".join(f"{system_tokens[l]:>12}" for l in LANGUAGES))
    print(f"{'complaint + framing':<20}" + "".join(f"{variable[l]:>12}" for l in LANGUAGES))
    print(f"{'system share':<20}" + "".join(
        f"{system_tokens[l] / request_tokens[l]:>11.1%}" for l in LANGUAGES))

    print(f"\nANNUAL BILL ON {model_key.upper()} -- US dollars per year")
    print("-" * 72)
    print(f"{'':<20}" + "".join(f"{lang.upper():>12}" for lang in LANGUAGES))
    uncached = {
        lang: cost_usd(model_key, request_tokens[lang], outputs[lang]) * per_year
        for lang in LANGUAGES
    }
    cached = {
        lang: cached_request_usd(model_key, system_tokens[lang], variable[lang],
                                 outputs[lang]) * per_year
        for lang in LANGUAGES
    }
    print(f"{'uncached (Part 3)':<20}" + "".join(f"{uncached[l]:>12,.0f}" for l in LANGUAGES))
    print(f"{'system prompt cached':<20}" + "".join(f"{cached[l]:>12,.0f}" for l in LANGUAGES))
    print(f"{'saved':<20}" + "".join(
        f"{uncached[l] - cached[l]:>12,.0f}" for l in LANGUAGES))
    print(f"{'saved %':<20}" + "".join(
        f"{1 - cached[l] / uncached[l]:>11.1%}" for l in LANGUAGES))

    print("\nTHE KAZAKH QUEUE -- the number this task exists for")
    print("-" * 72)
    kk_saved = uncached["kk"] - cached["kk"]
    print(f"KK uncached: ${uncached['kk']:,.0f}/year")
    print(f"KK with the system prompt cached: ${cached['kk']:,.0f}/year")
    print(f"Saved: ${kk_saved:,.0f}/year ({kk_saved / uncached['kk']:.1%}).")
    gap_before = uncached["kk"] - uncached["en"]
    gap_after = cached["kk"] - cached["en"]
    print(f"KK-over-EN gap: ${gap_before:,.0f} -> ${gap_after:,.0f}/year "
          f"({uncached['kk'] / uncached['en']:.2f}x -> {cached['kk'] / cached['en']:.2f}x) "
          "when both queues cache.")

    print("\nWHY THE SAVING IS THIS SIZE, AND NOT LARGER")
    print("-" * 72)
    model = MODELS[model_key]
    for lang in LANGUAGES:
        in_share = (request_tokens[lang] * model.input_per_mtok) / (
            request_tokens[lang] * model.input_per_mtok
            + outputs[lang] * model.output_per_mtok)
        print(f"  {lang.upper()}: input is {in_share:.1%} of the uncached bill; "
              f"the system prompt is {system_tokens[lang] / request_tokens[lang]:.1%} "
              "of that input.")
    print("  Caching can only touch that overlap. Output is priced at 5x input on\n"
          "  every model in prices.py and is not cacheable, which is why an\n"
          "  answer-length lever (task 6) moves this bill further than this one.")

    print("\nTWO THINGS THIS MODEL DOES NOT CHARGE FOR")
    print("-" * 72)
    per_hit_saving = {
        lang: (cost_usd(model_key, request_tokens[lang], outputs[lang])
               - cached_request_usd(model_key, system_tokens[lang], variable[lang],
                                    outputs[lang]))
        for lang in LANGUAGES
    }
    if provider == "anthropic":
        write_cost = {
            lang: system_tokens[lang] * MODELS[model_key].input_per_mtok
            * CACHE_WRITE_MULTIPLIER / 1_000_000
            for lang in LANGUAGES
        }
        print(f"1. The cache WRITE, at ~{CACHE_WRITE_MULTIPLIER}x the input rate, paid once per")
        print("   entry and again after every TTL lapse. Break-even, per write:")
        for lang in LANGUAGES:
            print(f"     {lang.upper()}: one write costs ${write_cost[lang]:.6f}; one hit saves "
                  f"${per_hit_saving[lang]:.6f} -> "
                  f"{write_cost[lang] / per_hit_saving[lang]:.1f} hits to break even.")
        print("   At the volume above that is seconds of traffic, so the lever stays\n"
              "   worth it -- but a queue idle longer than the TTL pays the write again.")
    else:
        # Gemini has two caching modes, not one. IMPLICIT caching is automatic
        # on 2.5 models and free to set up, but opportunistic: Google reuses a
        # matching prefix when it happens to still be around, with no client
        # control over the hit rate -- so every number above this line is an
        # optimistic upper bound, not a guarantee. EXPLICIT caching (a
        # CachedContent object you create yourself, with a TTL) guarantees the
        # hit but bills storage by the hour, on top of the discounted read.
        storage_rate = GEMINI_CACHE_STORAGE_PER_MTOK_HOUR[model_key]
        print("1. This model assumed every request hits the cache -- true for Gemini's")
        print("   IMPLICIT caching (automatic on 2.5 models, free, but opportunistic:")
        print("   no client-side control over when a prefix is actually reused).")
        print("   EXPLICIT caching (you create the cache yourself) guarantees the hit")
        print(f"   but bills storage at ${storage_rate:.2f}/Mtok/hour on top. Break-even,")
        print("   hits needed per hour of storage to justify explicit caching:")
        for lang in LANGUAGES:
            storage_cost_per_hour = system_tokens[lang] * storage_rate / 1_000_000
            hits_needed = storage_cost_per_hour / per_hit_saving[lang]
            print(f"     {lang.upper()}: storage costs ${storage_cost_per_hour:.6f}/hour; "
                  f"one hit saves ${per_hit_saving[lang]:.6f} -> "
                  f"{hits_needed:.1f} hits/hour to break even.")
        requests_per_hour = args.requests_per_day / 24
        print(f"   At {requests_per_hour:,.0f} requests/hour (from the volume above, spread")
        print("   evenly), explicit caching clears that easily -- but implicit caching,")
        print("   which costs nothing to try, may already capture most of this for free.")
    print(f"2. BATCH_DISCOUNT ({BATCH_DISCOUNT:.0%}) is NOT applied here. The Batch API is")
    print("   asynchronous, with a completion window in hours; a customer waiting in a")
    print("   live support queue cannot be served from it. Claiming it on this bill")
    print("   would be a fabricated saving. It is the right lever for offline work --")
    print("   nightly classification, bulk summarisation, eval runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())