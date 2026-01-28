from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from domain.project import Project
from domain.source_document import ManualTranscript, ASRExtract


PROJECT_EXT = ".ohsproj"
PROJECT_FILE = "project.json"
ASSETS_DIR = "assets"


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
        # fall back to directory
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

    # Keep original suffix if possible
    suffix = src.suffix if src.suffix else ""
    dest = assets_dir / f"{name_hint}{suffix}"

    # If file exists, overwrite (speichern ersetzt Stand)
    shutil.copy2(src, dest)
    return str(Path(ASSETS_DIR) / dest.name)  # relative path like "assets/transcript.odt"


def _project_to_dict(project):
    data: Dict[str, Any] = {
        "schema_version": 1,
        "project_id": str(project.project_id),
        "title": project.title,
        "description": project.description,
        "modified_at": _utc_now_iso(),
        "pipeline": project.preprocessing_pipeline,
        "preprocessing_results": project.preprocessing_results,
        "deviation_analysis_results": project.deviation_analysis_results,
        "alignment_results": project.alignment_results,
        "sources": {
            "transcript": None,
            "asr_extract": None,
        }
    }

    return data


def save_project(project: Project, project_dir):
    project_dir = _ensure_project_dir(Path(project_dir))
    assets_dir = project_dir / ASSETS_DIR

    transcript_path = project.transcript.file_name
    if transcript_path:
        t_path = Path(transcript_path)
    else:
        if getattr(project, "transcript_path", None):
            t_path = Path(getattr(project, "transcript_path", ""))
        else:
            t_path = None

    asr_extract_path = project.asr_extract.file_name
    if asr_extract_path:
        a_path = Path(asr_extract_path)
    else:
        if getattr(project, "asr_extract_path", None):
            a_path = Path(getattr(project, "asr_extract_path", ""))
        else:
            a_path = None

    data = _project_to_dict(project)

    # Copy assets
    if t_path is not None and t_path.exists():
        rel = _copy_asset(t_path, assets_dir, "transcript")
        data["sources"]["transcript"] = {"path": rel, "original_path": str(t_path)}
    else:
        data["sources"]["transcript"] = None

    if a_path is not None and a_path.exists():
        rel = _copy_asset(a_path, assets_dir, "asr_extract")
        data["sources"]["asr_extract"] = {"path": rel, "original_path": str(a_path)}
    else:
        data["sources"]["asr_extract"] = None

    # Write json
    project_file = project_dir / PROJECT_FILE
    with open(project_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return project_dir


def load_project(project_dir):
    """
    Loads project.json from a .ohsproj directory.
    """
    project_dir = Path(project_dir)
    project_file = project_dir / PROJECT_FILE

    with open(project_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    project = Project()
    project.project_id = uuid.UUID(data["project_id"])
    project.title = data.get("title", "")
    project.description = data.get("description", "")

    project.preprocessing_results = data.get("preprocessing_results", [])
    project.deviation_analysis_results = data.get("deviation_analysis_results", [])
    project.alignment_results = data.get("alignment_results", [])

    project.preprocessing_pipeline = None

    sources = data.get("sources", {})
    t_info = sources.get("transcript")
    a_info = sources.get("asr_extract")

    # Store paths on project for later saves
    # Resolve relative asset paths
    def _resolve_path(info):
        if not info:
            return None
        path_str = info.get("path")
        if not path_str:
            return None
        path = Path(path_str)
        if not path.is_absolute():
            path = project_dir / path
        return path

    t_path = _resolve_path(t_info)
    a_path = _resolve_path(a_info)

    setattr(project, "transcript_path", str(t_path) if t_path else None)
    setattr(project, "asr_extract_path", str(a_path) if a_path else None)

    project.transcript = ManualTranscript(str(t_path)) if t_path and t_path.exists() else None
    if a_path and a_path.exists():
        project.asr_extract = ASRExtract(str(a_path))
    else:
        project.asr_extract = None

    print("Test")

    return project
