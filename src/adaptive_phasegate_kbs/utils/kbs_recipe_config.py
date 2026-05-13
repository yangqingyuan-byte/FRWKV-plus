from __future__ import annotations

import json
from pathlib import Path


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "kbs_ours_recipes.json"
)


def _resolve_config_path(config_path=None) -> Path:
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    return path.resolve()


def load_recipe_config(config_path=None) -> dict:
    path = _resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"KBS recipe config not found: {path}")
    with open(path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"invalid KBS recipe config format: {path}")
    return data


def list_recipe_names(config_path=None) -> list[str]:
    data = load_recipe_config(config_path)
    return sorted(data.get("recipes", {}).keys())


def get_recipe_group(group_name: str, config_path=None) -> list[str]:
    data = load_recipe_config(config_path)
    try:
        group = data["recipe_groups"][group_name]
    except KeyError as exc:
        raise KeyError(f"unknown KBS recipe group: {group_name}") from exc
    if not isinstance(group, list):
        raise ValueError(f"recipe group must be a list: {group_name}")
    return list(group)


def get_seed_set(seed_set_name: str, config_path=None) -> list[int]:
    data = load_recipe_config(config_path)
    try:
        seed_set = data["seed_sets"][seed_set_name]
    except KeyError as exc:
        raise KeyError(f"unknown KBS seed set: {seed_set_name}") from exc
    if not isinstance(seed_set, list):
        raise ValueError(f"seed set must be a list: {seed_set_name}")
    return [int(item) for item in seed_set]


def get_model_group(group_name: str, config_path=None) -> list[str]:
    data = load_recipe_config(config_path)
    try:
        group = data["model_groups"][group_name]
    except KeyError as exc:
        raise KeyError(f"unknown KBS model group: {group_name}") from exc
    if not isinstance(group, list):
        raise ValueError(f"model group must be a list: {group_name}")
    return list(group)


def get_runtime_profile(profile_name=None, config_path=None) -> dict:
    data = load_recipe_config(config_path)
    if not profile_name:
        profile_name = data.get("default_runtime_profile", "ablation_stable")
    try:
        profile = data["runtime_profiles"][profile_name]
    except KeyError as exc:
        raise KeyError(f"unknown KBS runtime profile: {profile_name}") from exc
    if not isinstance(profile, dict):
        raise ValueError(f"runtime profile must be a dict: {profile_name}")
    return dict(profile)


def get_recipe_entry(recipe_name: str, config_path=None) -> dict:
    data = load_recipe_config(config_path)
    try:
        entry = data["recipes"][recipe_name]
    except KeyError as exc:
        raise KeyError(f"unknown KBS recipe: {recipe_name}") from exc
    if not isinstance(entry, dict):
        raise ValueError(f"recipe entry must be a dict: {recipe_name}")
    return dict(entry)


def resolve_recipe_args(recipe_name: str, config_path=None) -> dict:
    entry = get_recipe_entry(recipe_name, config_path=config_path)
    args = entry.get("args", entry)
    if not isinstance(args, dict):
        raise ValueError(f"recipe args must be a dict: {recipe_name}")
    return dict(args)


def resolve_train_args(recipe_name: str, *, overrides=None, runtime_profile=None, config_path=None) -> dict:
    profile = get_runtime_profile(runtime_profile, config_path=config_path)
    train_args = {}
    train_args.update(profile.get("train_args", {}))
    train_args.update(resolve_recipe_args(recipe_name, config_path=config_path))
    if overrides:
        train_args.update({key: value for key, value in overrides.items() if value is not None})
    return train_args
