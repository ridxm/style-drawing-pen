"""Generate drawing stroke control points from a text prompt via GPT-4o."""

from __future__ import annotations

import json
import os
from typing import List, Tuple

from openai import OpenAI

Stroke = List[Tuple[float, float]]

SYSTEM_PROMPT = (
    "You are a drawing instruction generator. Given a text description, "
    "return a JSON object with a 'strokes' array. Each stroke is an array "
    "of [x,y] control points in a 500x500 coordinate space. Keep drawings "
    "simple but recognizable: 5-20 strokes, 3-8 control points per stroke. "
    "Group strokes logically (outline first, then details). Return ONLY "
    "valid JSON, no markdown, no explanation."
)

FALLBACK_SHAPES: dict[str, dict] = {
    "house": {
        "strokes": [
            [[100, 400], [100, 250], [250, 150], [400, 250], [400, 400], [100, 400]],
            [[180, 400], [180, 320], [260, 320], [260, 400]],
            [[300, 300], [360, 300], [360, 250], [300, 250], [300, 300]],
            [[100, 250], [250, 150], [400, 250]],
        ]
    },
    "tree": {
        "strokes": [
            [[240, 450], [240, 330], [260, 330], [260, 450]],
            [[250, 330], [170, 300], [150, 240], [200, 200], [250, 170],
             [300, 200], [350, 240], [330, 300], [250, 330]],
            [[220, 260], [240, 250], [260, 260]],
            [[270, 230], [290, 220], [310, 230]],
        ]
    },
    "cat": {
        "strokes": [
            [[180, 250], [180, 350], [320, 350], [320, 250], [180, 250]],
            [[180, 250], [160, 200], [210, 240]],
            [[320, 250], [340, 200], [290, 240]],
            [[220, 290], [230, 290]],
            [[270, 290], [280, 290]],
            [[240, 320], [250, 330], [260, 320]],
            [[200, 330], [170, 340]],
            [[200, 340], [170, 345]],
            [[300, 330], [330, 340]],
            [[300, 340], [330, 345]],
        ]
    },
    "star": {
        "strokes": [
            [[250, 100], [290, 210], [400, 210], [315, 280],
             [345, 390], [250, 325], [155, 390], [185, 280],
             [100, 210], [210, 210], [250, 100]],
        ]
    },
    "face": {
        "strokes": [
            [[250, 100], [370, 160], [400, 280], [340, 390],
             [250, 420], [160, 390], [100, 280], [130, 160], [250, 100]],
            [[190, 230], [200, 220], [220, 220], [230, 230]],
            [[270, 230], [280, 220], [300, 220], [310, 230]],
            [[200, 240], [210, 250]],
            [[290, 240], [300, 250]],
            [[240, 260], [245, 300], [255, 300], [260, 260]],
            [[200, 340], [240, 360], [280, 360], [310, 340]],
        ]
    },
}


def _strokes_from_json(data: dict) -> List[Stroke]:
    strokes_raw = data["strokes"]
    result: List[Stroke] = []
    for stroke in strokes_raw:
        result.append([(float(pt[0]), float(pt[1])) for pt in stroke])
    return result


def _fallback(prompt: str) -> List[Stroke]:
    key = prompt.lower()
    for name, shape in FALLBACK_SHAPES.items():
        if name in key:
            return _strokes_from_json(shape)
    return _strokes_from_json(FALLBACK_SHAPES["star"])


def generate(prompt: str) -> List[Stroke]:
    """Return strokes as list[list[(x,y)]] in a 500x500 space."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback(prompt)
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=15,
        )
        content = resp.choices[0].message.content or ""
        data = json.loads(content)
        return _strokes_from_json(data)
    except Exception:
        return _fallback(prompt)


if __name__ == "__main__":
    strokes = generate("a simple cat")
    for i, stroke in enumerate(strokes):
        print(f"stroke {i}: {stroke}")
