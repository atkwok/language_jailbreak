"""
Utility helpers to translate jailbreak prompts to other languages for evaluation.

Replace the `translate_text` stub with a real translation call (e.g., an API or local model)
and expand the main block with the languages and models you want to test.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List

DATA_DIR = Path(__file__).resolve().parent / "data"
TRANSLATIONS_DIR = Path(__file__).resolve().parent / "translations"
LANGUAGES = ("es", "fr")

def load_prompts(prompts_path: Path = DATA_DIR / "prompts.en.json") -> List[str]:
    """Load the English jailbreak prompts from disk."""
    with prompts_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("prompts", [])


def translate_text(text: str, target_lang: str) -> str:
    """
    Placeholder translation hook.

    Swap this out for a real translation provider or model. For now, it tags the text
    with the target language so downstream steps have a predictable shape.
    """
    pass 
    return "TODO translated text"


def translate_prompts(target_lang: str, prompts: Iterable[str]) -> Dict[str, object]:
    """Translate a collection of prompts into a target language."""
    translated = [translate_text(prompt, target_lang) for prompt in prompts]
    return {
        "language": target_lang,
        "source_language": "en",
        "prompts": translated,
    }


def save_translations(payload: Dict[str, object], target_lang: str) -> Path:
    """Persist translated prompts to the translations directory."""
    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRANSLATIONS_DIR / f"prompts.{target_lang}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output_path


if __name__ == "__main__":
    prompts = load_prompts()
    for lang in LANGUAGES:
        payload = translate_prompts(lang, prompts)
        dest = save_translations(payload, lang)
        print(f"Saved translations to {dest}")
