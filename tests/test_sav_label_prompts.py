"""Tests for the descriptive SAV label prompt bank."""

from trokens.models.pointformer import Pointformer


SAV_ATOMIC_LABELS = [
    "sit",
    "stand",
    "look_forward",
    "look_sideways",
    "read",
    "flip_books",
    "touch_sth",
    "raise_hand",
    "hands_down",
    "take_notes",
    "applaud",
    "bend",
    "turn_around",
    "talk_with_others",
    "answer_questions",
]


def test_every_sav_label_has_a_descriptive_prompt_group():
    model = Pointformer.__new__(Pointformer)

    for label_name in SAV_ATOMIC_LABELS:
        prompts = model._get_sav_label_prompts(label_name)

        assert len(prompts) == 5
        assert prompts[0]
        assert "_" not in prompts[0]
        assert len(set(prompts)) == len(prompts)


def test_unknown_sav_label_keeps_readable_fallback_prompt():
    model = Pointformer.__new__(Pointformer)

    assert model._get_sav_label_prompts("new_action") == ["new action"]
