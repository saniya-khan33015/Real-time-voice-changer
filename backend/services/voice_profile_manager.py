import json
import shutil

from backend.core.config import get_settings
from backend.utils.audio import safe_name


def _profile_dir(name: str):
    return get_settings().cloned_voices_dir / safe_name(name)


def list_profiles() -> list[dict]:
    root = get_settings().cloned_voices_dir
    root.mkdir(parents=True, exist_ok=True)
    profiles = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = load_profile(folder.name)
        if "error" not in metadata:
            profiles.append(metadata)
    return profiles


def save_profile(name: str, data: str | dict) -> dict:
    folder = _profile_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    payload = json.loads(data) if isinstance(data, str) else data
    payload.setdefault("name", safe_name(name))
    with (folder / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return {"status": "saved", "profile": payload}


def rename_profile(old_name: str, new_name: str) -> dict:
    old_dir = _profile_dir(old_name)
    new_dir = _profile_dir(new_name)
    if not old_dir.exists():
        return {"error": "Profile not found"}
    if new_dir.exists():
        return {"error": "Target profile already exists"}
    old_dir.rename(new_dir)
    metadata = load_profile(new_name)
    if "error" not in metadata:
        metadata["name"] = safe_name(new_name)
        metadata["display_name"] = new_name
        save_profile(new_name, metadata)
    return {"status": "renamed"}


def delete_profile(name: str) -> dict:
    folder = _profile_dir(name)
    if not folder.exists():
        return {"error": "Profile not found"}
    shutil.rmtree(folder)
    return {"status": "deleted"}


def load_profile(name: str) -> dict:
    folder = _profile_dir(name)
    metadata_path = folder / "metadata.json"
    if not metadata_path.exists():
        return {"error": "Profile not found"}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
