"""Optional Claude judge: reads the transcript of the last couple of minutes and
returns the moments that are actually funny. Used only when Anthropic credentials
are available (ANTHROPIC_API_KEY, or a profile from `ant auth login`)."""
from __future__ import annotations

import json
import os

MODEL = os.environ.get("CLIPPER_JUDGE_MODEL", "claude-opus-5-5")

SCHEMA = {
    "type": "object",
    "properties": {
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "score": {"type": "integer"},
                    "title": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["start", "end", "score", "title", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["moments"],
    "additionalProperties": False,
}

SYSTEM = """You pick clips for a channel that posts the funniest moments of live streams as short videos.

You get the transcript of the last two minutes or so of a stream, with start and end seconds on every line, plus the times of loud audio bursts (often laughter or shouting) and of viewers laughing in live chat. Speech recognition makes mistakes and the transcript may be in any language.

Return the moments a viewer would actually laugh at: a joke that lands, a fail, a ridiculous reaction, an absurd exchange. Each moment must stand on its own, so start it where the setup begins and end it just after the punchline or reaction, 12 to 45 seconds long. Most windows contain nothing worth clipping; return an empty list for those rather than a weak pick.

For each moment give: start and end in seconds (using the transcript's clock), a score from 1 to 10 for how funny it is (7 and up means you would post it), a short catchy title in the language of the stream, and a few words on why it is funny."""


def make_judge():
    """Returns a client, or None when no credentials are configured."""
    import anthropic
    try:
        client = anthropic.Anthropic()
        client.models.retrieve(MODEL)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.APIConnectionError):
        return None
    except anthropic.APIStatusError:
        return client          # credentials work; the model lookup itself is not essential
    except Exception:
        return None            # no credentials configured
    return client


def find_moments(client, title, lines, lo, hi, win_end, loud_spikes, chat_spikes) -> list[dict]:
    user = (
        f"Stream title: {title}\n\n"
        f"Transcript (seconds 0 to {win_end:.0f}):\n" + "\n".join(lines) + "\n\n"
        f"Loud audio bursts at: {', '.join(loud_spikes) or 'none'}\n"
        f"Chat laughing at: {', '.join(chat_spikes) or 'none'}\n\n"
        f"Only return moments whose middle falls between {lo:.0f}s and {hi:.0f}s; the rest of "
        f"the transcript is context that is covered by the neighbouring windows."
    )
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=8000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if resp.stop_reason == "refusal":
        return []
    text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(text).get("moments", [])
