from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from domain.project import Project, PreprocessingResultCollection
from domain.source_document import ManualTranscript, ASRExtract


PROJECT_EXT = ".ohsproj"
PROJECT_FILE = "project.json"
ASSETS_DIR = "assets"
PREPROCESSING_DIR = "preprocessing"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_project_dir(target: Path) -> Path:
    """
    Accepts either a directory path or a file-like path ending with .ohsproj.
    Creates the directory if missing and returns it.
    """
    target = Path(target)

    # Allow user to pass ".../MyProject" and we add extension
    if target.suffix == "":
        target = target.with_suffix(PROJECT_EXT)

    # If they pass something else, still treat it as directory
    if target.suffix != PROJECT_EXT:
        target.mkdir(parents=True, exist_ok=True)
        return target

    target.mkdir(parents=True, exist_ok=True)
    return target

def _copy_asset(src: Path, assets_dir: Path, name_hint: str) -> str:
    """
    Copy src into assets_dir with a stable name. Returns relative path string.
    """
    src = Path(src)
    assets_dir.mkdir(parents=True, exist_ok=True)

    suffix = src.suffix if src.suffix else ""
    dest = assets_dir / f"{name_hint}{suffix}"

    # overwrite existing file
    shutil.copy2(src, dest)
    return str(Path(ASSETS_DIR) / dest.name)  # e.g. "assets/transcript.odt"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_asset_path(project_dir: Path, info: Optional[Dict[str, Any]]) -> Optional[Path]:
    if not info:
        return None
    path_str = info.get("path")
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = project_dir / p
    return p


def _write_preprocessing_artifact(assets_dir: Path,preprocessing_payload: Any) -> Optional[Dict[str, Any]]:
    """
    Stores preprocessing payload in assets/preprocessing/preprocessing_results.prep.json
    and returns metadata dict to embed into project.json.
    """
    if preprocessing_payload is None:
        return None

    prep_dir = assets_dir / PREPROCESSING_DIR
    prep_file = prep_dir / "preprocessing_results.prep.json"

    _write_json(prep_file, preprocessing_payload)

    rel_path = str(Path(ASSETS_DIR) / PREPROCESSING_DIR / prep_file.name)

    return {
        "path": rel_path,
        "created_at": _utc_now_iso(),
        "state": "valid",
    }


def _read_preprocessing_artifact(project_dir: Path, info: Optional[Dict[str, Any]]) -> Optional[Any]:
    if not info:
        return None
    path_str = info.get("path")
    if not path_str:
        return None

    p = Path(path_str)
    if not p.is_absolute():
        p = project_dir / p

    if not p.exists():
        return None

    return _read_json(p)


def save_project(project: Project, project_dir) -> Path:
    project_dir = _ensure_project_dir(Path(project_dir))
    assets_dir = project_dir / ASSETS_DIR

    # Build base json from domain object
    data = project.to_dict()
    data.setdefault("sources", {})

    # preprocessing reference
    data["preprocessing"] = None

    # ------ Copy transcript asset --------
    transcript_path_str = getattr(project.transcript, "file_name", None) if project.transcript else None
    t_path = Path(transcript_path_str) if transcript_path_str else None

    if t_path and t_path.exists():
        rel = _copy_asset(t_path, assets_dir, "transcript")
        data["sources"]["transcript"] = {"path": rel, "original_path": str(t_path)}
    else:
        data["sources"]["transcript"] = None

    # ------ Copy ASR extract asset --------
    asr_path_str = getattr(project.asr_extract, "file_name", None) if project.asr_extract else None
    a_path = Path(asr_path_str) if asr_path_str else None

    if a_path and a_path.exists():
        rel = _copy_asset(a_path, assets_dir, "asr_extract")
        data["sources"]["asr_extract"] = {"path": rel, "original_path": str(a_path)}
    else:
        data["sources"]["asr_extract"] = None

    # --- Preprocessing: externalize ---
    # Convert collections to JSON payload
    preprocessing_payload = [p.to_dict() for p in getattr(project, "preprocessing_results", [])]

    data["preprocessing"] = _write_preprocessing_artifact(assets_dir=assets_dir, preprocessing_payload=preprocessing_payload)

    # Remove the heavy inline list from project.json (keeps JSON small)
    # For backward compatibility, we keep the key but empty it.
    data["preprocessing_results"] = []

    # Write project.json
    project_file = project_dir / PROJECT_FILE
    _write_json(project_file, data)

    return project_dir

def load_project(project_dir) -> Project:
    project_dir = Path(project_dir)
    project_file = project_dir / PROJECT_FILE

    data = _read_json(project_file)
    project = Project.from_dict(data)

    sources = data.get("sources", {})
    t_info = sources.get("transcript")
    a_info = sources.get("asr_extract")

    t_path = _resolve_asset_path(project_dir, t_info)
    a_path = _resolve_asset_path(project_dir, a_info)

    # Store paths on project for later saves
    setattr(project, "transcript_path", str(t_path) if t_path else None)
    setattr(project, "asr_extract_path", str(a_path) if a_path else None)

    # Recreate source docs from copied assets
    project.transcript = ManualTranscript(str(t_path)) if t_path and t_path.exists() else None
    project.asr_extract = ASRExtract(str(a_path)) if a_path and a_path.exists() else None

    # --- Load preprocessing (new format) ---
    prep_info = data.get("preprocessing")
    prep_payload = _read_preprocessing_artifact(project_dir, prep_info)

    if prep_payload is not None:
        project.preprocessing_results = [PreprocessingResultCollection.from_dict(x) for x in prep_payload]
        project.current.preprocessing = project.preprocessing_results[-1]
    else:
        # --- Fallback: old projects storing preprocessing inline ---
        project.preprocessing_results = [
            PreprocessingResultCollection.from_dict(p) for p in data.get("preprocessing_results", [])
        ]

    return project
