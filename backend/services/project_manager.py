import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from backend.core.config import get_settings
from backend.pipeline.xtts_pipeline import generate_voice
from backend.services.tts import generate_speech_sync
from backend.utils.audio import concatenate_wavs, export_audio, safe_name


def parse_script(script: str) -> list[dict]:
    lines = []
    current_speaker = "Narrator"
    for raw in script.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            speaker, text = line.split(":", 1)
            current_speaker = speaker.strip() or current_speaker
            text = text.strip()
        else:
            text = line
        if text:
            lines.append({"speaker": current_speaker, "text": text})
    return lines


def _project_dir(name: str) -> Path:
    return get_settings().projects_dir / safe_name(name)


def _project_file(name: str) -> Path:
    return _project_dir(name) / "project.json"


def create_project(name: str, script: str, speakers: str | dict) -> dict:
    speaker_map = json.loads(speakers) if isinstance(speakers, str) else speakers
    folder = _project_dir(name)
    (folder / "generated_audio").mkdir(parents=True, exist_ok=True)
    project = {
        "name": safe_name(name),
        "display_name": name,
        "script": script,
        "speakers": speaker_map,
        "lines": parse_script(script),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_project(name, project)
    (folder / "script.txt").write_text(script, encoding="utf-8")
    (folder / "voices.json").write_text(json.dumps(speaker_map, indent=2), encoding="utf-8")
    return {"status": "created", "project": project}


def list_projects() -> list[dict]:
    root = get_settings().projects_dir
    root.mkdir(parents=True, exist_ok=True)
    projects = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        project = load_project(folder.name)
        if "error" not in project:
            projects.append({"name": project["name"], "display_name": project.get("display_name", project["name"])})
    return projects


def load_project(name: str) -> dict:
    path = _project_file(name)
    if not path.exists():
        return {"error": "Project not found"}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_project(name: str, data: str | dict) -> dict:
    project = json.loads(data) if isinstance(data, str) else data
    project.setdefault("name", safe_name(name))
    project["updated_at"] = datetime.now(timezone.utc).isoformat()
    folder = _project_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    with _project_file(name).open("w", encoding="utf-8") as handle:
        json.dump(project, handle, indent=2)
    return {"status": "saved", "project": project}


def delete_project(name: str) -> dict:
    folder = _project_dir(name)
    if not folder.exists():
        return {"error": "Project not found"}
    shutil.rmtree(folder)
    return {"status": "deleted"}


def generate_project_audio(name: str, pause_seconds: float = 0.35) -> dict:
    project = load_project(name)
    if "error" in project:
        return project
    folder = _project_dir(name)
    generated_dir = folder / "generated_audio"
    generated_dir.mkdir(parents=True, exist_ok=True)
    segment_paths = []
    timeline = []
    for index, line in enumerate(project.get("lines") or parse_script(project.get("script", "")), start=1):
        speaker = line["speaker"]
        reference_voice = project.get("speakers", {}).get(speaker)
        out_path = generated_dir / f"{index:03d}_{safe_name(speaker)}.wav"
        if reference_voice and Path(reference_voice).exists():
            result = generate_voice(line["text"], reference_voice)
            generated = result["audio_path"]
            voice_label = reference_voice
        else:
            voice_profile = reference_voice or speaker
            generated = generate_speech_sync(line["text"], voice_profile)
            voice_label = voice_profile
        shutil.copyfile(generated, out_path)
        segment_paths.append(str(out_path))
        timeline.append({"index": index, "speaker": speaker, "voice": voice_label, "text": line["text"], "audio": str(out_path)})
    final_mix = folder / "final_mix.wav"
    concatenate_wavs(segment_paths, str(final_mix), pause_seconds=pause_seconds)
    project["timeline"] = timeline
    project["final_mix"] = str(final_mix)
    save_project(name, project)
    return {"status": "generated", "audio_path": str(final_mix), "timeline": timeline}


def export_project(name: str, fmt: str = "wav") -> dict:
    project = load_project(name)
    if "error" in project:
        return project
    final_mix = project.get("final_mix")
    if not final_mix or not Path(final_mix).exists():
        result = generate_project_audio(name)
        if "error" in result:
            return result
        final_mix = result["audio_path"]
    output_path = _project_dir(name) / f"{safe_name(name)}_export.{fmt.lower()}"
    return {"status": "exported", "audio_path": export_audio(final_mix, str(output_path))}
