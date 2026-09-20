"""Claude API list prices, in US dollars per million tokens.

Prices change. This table is a snapshot, not a fact of nature -- check it
against the source below before you quote a number to anyone.

Source: https://platform.claude.com/docs/en/about-claude/models/overview.md
Checked: 5 September 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

PRICE_SOURCE = "https://platform.claude.com/docs/en/about-claude/models/overview.md"
PRICE_CHECKED = "2026-09-12"

#: Batch API requests are billed at half the synchronous price.
BATCH_DISCOUNT = 0.50

#: A prompt-cache read costs 10% of the base input price on opus-5, sonnet-5
#: and haiku-4.5. It is NOT uniform: on fable-5.1 a cache read is 2.5% of the
#: input price ($0.25/Mtok against $10.00/Mtok). Checked 2026-09-06.
CACHE_READ_FRACTION = {
    "fable-5.1": 0.025,
    "opus-5": 0.10,
    "sonnet-5": 0.10,
    "haiku-4.5": 0.10,
}


@dataclass(frozen=True)
class Model:
    """One model's list price and headline limits.

    Attributes:
        model_id: The exact string sent as ``model`` in an API request.
        input_per_mtok: US dollars per one million input tokens.
        output_per_mtok: US dollars per one million output tokens.
        context_tokens: Size of the context window, in tokens.
    """

    model_id: str
    input_per_mtok: float
    output_per_mtok: float
    context_tokens: int


MODELS: Dict[str, Model] = {
    "fable-5.1": Model("claude-fable-5-1", 10.00, 50.00, 1_000_000),
    "opus-5": Model("claude-opus-5", 5.00, 25.00, 1_000_000),
    "sonnet-5": Model("claude-sonnet-5", 2.00, 10.00, 1_000_000),
    "haiku-4.5": Model("claude-haiku-4-5", 1.00, 5.00, 200_000),
}

DEFAULT_MODEL = "opus-5"

#: Order to print the Claude price tables in, cheapest first.
CLAUDE_MODEL_ORDER = ("haiku-4.5", "sonnet-5", "opus-5", "fable-5.1")

#: Gemini Developer API list prices, in US dollars per million tokens.
#: gemini-2.5-pro's rate below is the <=200k-token tier; it doubles on input
#: over 200k tokens and rises to $15/Mtok on output -- irrelevant here since
#: this lab's texts are a few hundred tokens, but not a fact to carry to a
#: larger prompt without checking the source again.
#:
#: Source: https://ai.google.dev/gemini-api/docs/pricing
#: Checked: 2026-09-20
GEMINI_PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
GEMINI_PRICE_CHECKED = "2026-09-20"

#: gemini-2.5-flash was retired for new API keys in September 2026; Google's
#: 404 response for new accounts points to gemini-3.6-flash. Price below is
#: UNVERIFIED against an official Google page as of this writing -- web
#: search returned only third-party pricing pages quoting $0.75/$3.75, not
#: ai.google.dev itself. CONFIRM THIS NUMBER at
#: https://ai.google.dev/gemini-api/docs/pricing before quoting it in a
#: report.
GEMINI_MODELS: Dict[str, Model] = {
    "gemini-2.5-flash": Model("gemini-2.5-flash", 0.075, 0.30, 1_048_576),
    "gemini-2.5-pro": Model("gemini-2.5-pro", 1.25, 10.00, 1_048_576),
    "gemini-3.6-flash": Model("gemini-3.6-flash", 0.75, 3.75, 1_048_576),  # UNVERIFIED, see note above
}

#: cost_usd() and --model on both part2/part3 scripts look models up in one
#: table, whichever provider they came from.
MODELS.update(GEMINI_MODELS)

#: Order to print the Gemini price tables in, cheapest first.
GEMINI_MODEL_ORDER = ("gemini-2.5-flash", "gemini-3.6-flash", "gemini-2.5-pro")

#: A Gemini cached-content read costs 10% of the base input price -- same
#: figure as Claude's opus-5/sonnet-5/haiku-4.5, coincidentally, not a
#: Google-Anthropic convention, just how it happens to land this quarter.
CACHE_READ_FRACTION.update({
    "gemini-2.5-flash": 0.10,
    "gemini-2.5-pro": 0.10,
    "gemini-3.6-flash": 0.10,  # ASSUMED same 10% pattern; unverified for 3.6
})

#: Gemini's EXPLICIT cache (a CachedContent object you create yourself, with
#: a TTL) bills storage by the hour on top of the discounted read -- a cost
#: Claude's cache has no equivalent line for. IMPLICIT caching (automatic,
#: on by default for 2.5 models, no code required) has no storage fee, but
#: is opportunistic: Google applies the discount when it happens to reuse a
#: previous prefix, with no guarantee and no client-side control over when.
#: Source: https://ai.google.dev/gemini-api/docs/pricing (context caching
#: storage). Checked 2026-09-20.
GEMINI_CACHE_STORAGE_PER_MTOK_HOUR = {
    "gemini-2.5-flash": 0.50,
    "gemini-2.5-pro": 4.50,
    "gemini-3.6-flash": 0.50,  # ASSUMED same as 2.5-flash; unverified for 3.6
}


def cost_usd(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """Return the list-price cost of one request, in US dollars.

    Args:
        model_key: A key of :data:`MODELS`, e.g. ``"opus-5"``.
        input_tokens: Tokens sent, including the system prompt and any history.
        output_tokens: Tokens generated by the model.

    Returns:
        Cost in US dollars. No caching, no batch discount, no free tier.

    Raises:
        KeyError: If ``model_key`` is not in :data:`MODELS`.
    """
    model = MODELS[model_key]
    return (
        input_tokens * model.input_per_mtok
        + output_tokens * model.output_per_mtok
    ) / 1_000_000