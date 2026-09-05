"""Tests for dataset-configured label prompts."""

from trokens.config.defaults import get_cfg
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


def _prompt_groups(config_path):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    model = Pointformer.__new__(Pointformer)
    model.cfg = cfg
    model.num_classes = cfg.MODEL.NUM_CLASSES
    model.atomic_label_names = model._load_atomic_label_names()
    return model.atomic_label_names, model._load_label_prompt_groups()


def test_every_sav_label_has_a_configured_descriptive_prompt_group():
    label_names, prompt_groups = _prompt_groups("configs/trokens/sav.yaml")

    assert label_names == SAV_ATOMIC_LABELS
    for prompts in prompt_groups:

        assert len(prompts) == 5
        assert prompts[0]
        assert "_" not in prompts[0]
        assert len(set(prompts)) == len(prompts)


def test_tinyvirat_uses_exact_class_names_without_sav_prompts():
    label_names, prompt_groups = _prompt_groups("configs/trokens/tinyvirat.yaml")

    assert prompt_groups == [[label_name] for label_name in label_names]
    assert prompt_groups[3] == ["activity_carrying"]
    assert prompt_groups[18] == ["specialized_talking_phone"]
