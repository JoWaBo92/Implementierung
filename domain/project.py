import uuid

from dataclasses import dataclass

import numpy as np

from spacy.tokens import Doc
import spacy

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from domain.deviation import DeviationAnalysisConfig
from domain.preprocessing import PreprocessingConfig, PreprocessingResult
from domain.source_document import ManualTranscript, ASRExtract
from domain.synchronization import SynchronizationConfig

class PreprocessingResultCollection:
    def __init__(self, config = PreprocessingConfig()):
        self.result_id = uuid.uuid4()
        self.time = datetime.now()
        self.transcript_results: List[PreprocessingResult] = []
        self.extract_results: List[PreprocessingResult] = []
        self.config: PreprocessingConfig = config

    def to_dict(self):
        return {
            "result_id": str(self.result_id),
            "time": self.time.isoformat(), 
            "transcript_results": [r.to_dict() for r in self.transcript_results],
            "extract_results": [r.to_dict() for r in self.extract_results],
            "config": self.config.to_dict()
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessingResultCollection":
        cfg = PreprocessingConfig.from_dict(d.get("config", {}))
        obj = cls(config=cfg)
        rid = d.get("result_id")
        obj.result_id = uuid.UUID(rid) if rid else uuid.uuid4()
        t = d.get("time")
        if t:
            obj.time = datetime.fromisoformat(t)
        nlp = spacy.load(cfg.spacy_model)
        obj.transcript_results = [PreprocessingResult.from_dict(x, nlp) for x in d.get("transcript_results", [])]
        obj.extract_results = [PreprocessingResult.from_dict(x, nlp) for x in d.get("extract_results", [])]
        return obj
    
class DeviationResultCollection:
    def __init__(self, config = DeviationAnalysisConfig()):
        self.result_id = uuid.uuid4()
        self.time = datetime.now()
        self.config: DeviationAnalysisConfig = config
        self.preprocessing_id: Optional[uuid.UUID] = None
        
        self.transcript_preprocessed: List[PreprocessingResult] = []
        self.extract_preprocessed: List[PreprocessingResult] = []
        self.result_matrix: np.ndarray = None

    def to_dict(self) -> dict:
        return {
            "result_id": str(self.result_id),
            "time": self.time.isoformat(),
            "config": self.config.to_dict(),
            "preprocessing_id": str(self.preprocessing_id) if self.preprocessing_id else None,
            "result_matrix": self.result_matrix.tolist() if isinstance(self.result_matrix, np.ndarray) else None
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeviationResultCollection":
        cfg = DeviationAnalysisConfig.from_dict(d.get("config")) if isinstance(d.get("config"), dict) else DeviationAnalysisConfig()
        obj = cls(config=cfg)
        rid = d.get("result_id")
        obj.result_id = uuid.UUID(rid) if rid else uuid.uuid4()
        t = d.get("time")
        if t:
            obj.time = datetime.fromisoformat(t)
        pid = d.get("preprocessing_id")
        obj.preprocessing_id = uuid.UUID(pid) if pid else None
        m = d.get("result_matrix")
        obj.result_matrix = np.array(m, dtype=float) if m is not None else None
        return obj

@dataclass
class AlignedSegment:
    transcript_index: int
    extract_j0: int
    extract_j1: int
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    mean_similarity: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "transcript_index": self.transcript_index,
            "extract_j0": self.extract_j0,
            "extract_j1": self.extract_j1,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "mean_similarity": self.mean_similarity
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AlignedSegment":
        return cls(
            transcript_index=int(d.get("transcript_index", 0)),
            extract_j0=int(d.get("extract_j0", -1)),
            extract_j1=int(d.get("extract_j1", -1)),
            start_ms=d.get("start_ms"),
            end_ms=d.get("end_ms"),
            mean_similarity=d.get("mean_similarity")
        )

class SynchronizationResultCollection:
    def __init__(
        self,
        config: SynchronizationConfig = SynchronizationConfig(),
        transcript_preprocessed: Optional[List[PreprocessingResult]] = None,
        extract_preprocessed: Optional[List[PreprocessingResult]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        alignment_path: Optional[List[Tuple[int, int]]] = None,
        alignment_ranges_by_transcript: Optional[List[Tuple[int, int]]] = None,
        total_cost: Optional[float] = None,
        mean_similarity_on_path: Optional[float] = None,
        aligned_transcript: Optional[List[AlignedSegment]] = None
    ):
        self.result_id = uuid.uuid4()
        self.time = datetime.now()
        self.config = config
        self.deviation_id: Optional[uuid.UUID] = None
        self.transcript_preprocessed = transcript_preprocessed or []
        self.extract_preprocessed = extract_preprocessed or []
        self.similarity_matrix = similarity_matrix
        self.alignment_path = alignment_path or []
        self.alignment_ranges_by_transcript = alignment_ranges_by_transcript or []
        self.total_cost = total_cost
        self.mean_similarity_on_path = mean_similarity_on_path
        self.aligned_transcript: List[AlignedSegment] = aligned_transcript or []

    def to_dict(self) -> dict:
        return {
            "result_id": str(self.result_id),
            "time": self.time.isoformat(),
            "config": self.config.to_dict(),
            "deviation_id": str(self.deviation_id) if self.deviation_id else None,
            "alignment_path": [list(p) for p in self.alignment_path],
            "alignment_ranges_by_transcript": [list(r) for r in self.alignment_ranges_by_transcript],
            "total_cost": self.total_cost,
            "mean_similarity_on_path": self.mean_similarity_on_path,
            "aligned_transcript": [a.to_dict() for a in (self.aligned_transcript or [])],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SynchronizationResultCollection":
        cfg = SynchronizationConfig.from_dict(d.get("config", {})) if hasattr(SynchronizationConfig, "from_dict") else SynchronizationConfig()
        obj = cls(config=cfg)
        rid = d.get("result_id")
        obj.result_id = uuid.UUID(rid) if rid else uuid.uuid4()
        t = d.get("time")
        if t:
            obj.time = datetime.fromisoformat(t)
        did = d.get("deviation_id")
        obj.deviation_id = uuid.UUID(did) if did else None

        obj.alignment_path = [tuple(p) for p in d.get("alignment_path", [])]
        obj.alignment_ranges_by_transcript = [tuple(r) for r in d.get("alignment_ranges_by_transcript", [])]
        obj.total_cost = d.get("total_cost")
        obj.mean_similarity_on_path = d.get("mean_similarity_on_path")
        obj.aligned_transcript = [AlignedSegment.from_dict(x) for x in d.get("aligned_transcript", [])]
        return obj

    def get_extract_range_for_transcript(self, transcript_index: int) -> Tuple[int, int]:
        if (
            transcript_index < 0
            or transcript_index >= len(self.alignment_ranges_by_transcript)
        ):
            return -1, -1
        return self.alignment_ranges_by_transcript[transcript_index]

    def get_aligned_extract_indices(self, transcript_index: int) -> List[int]:
        j0, j1 = self.get_extract_range_for_transcript(transcript_index)
        if j0 < 0 or j1 < 0:
            return []
        return list(range(j0, j1 + 1))

    def mean_similarity_for_transcript(self, transcript_index: int) -> Optional[float]:
        if self.similarity_matrix is None:
            return None
        j0, j1 = self.get_extract_range_for_transcript(transcript_index)
        if j0 < 0 or j1 < 0:
            return None
        return float(np.mean(self.similarity_matrix[transcript_index, j0:j1 + 1]))
    
    def build_aligned_transcript(self, extract_times_ms: List[int]) -> None:
        self.aligned_transcript = []

        nE = len(extract_times_ms)
        sim = self.similarity_matrix

        for i in range(len(self.transcript_preprocessed)):
            j0, j1 = self.get_extract_range_for_transcript(i)

            if j0 < 0 or j1 < 0 or j0 > j1 or j0 >= nE:
                self.aligned_transcript.append(
                    AlignedSegment(transcript_index=i, extract_j0=j0, extract_j1=j1)
                )
                continue

            j0 = max(0, int(j0))
            j1 = min(int(j1), nE - 1)

            start_ms = int(extract_times_ms[j0])
            if j1 + 1 < nE:
                end_ms = int(extract_times_ms[j1 + 1])
            else:
                end_ms = int(extract_times_ms[j1])

            mean_sim = None
            if isinstance(sim, np.ndarray) and sim.ndim == 2:
                if i < sim.shape[0] and j1 < sim.shape[1]:
                    row = sim[i, j0:j1 + 1]
                    finite = np.isfinite(row)
                    if np.any(finite):
                        mean_sim = float(np.mean(row[finite]))

            self.aligned_transcript.append(
                AlignedSegment(
                    transcript_index=i,
                    extract_j0=j0,
                    extract_j1=j1,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    mean_similarity=mean_sim,
                )
            )

@dataclass
class ProjectCurrentState:
    preprocessing: Optional[PreprocessingResultCollection] = None
    deviation_analysis: Optional[DeviationResultCollection] = None
    synchronization: Optional[SynchronizationResultCollection] = None

class Project:
    def __init__(self, title: str = "", description: str = "", transcript: ManualTranscript = None, asr_extract: ASRExtract = None):
        self.project_id = uuid.uuid4()
        self.title: str = title
        self.description: str = description
        self.transcript: ManualTranscript = transcript
        self.asr_extract: ASRExtract = asr_extract
        self.preprocessing_results: List[PreprocessingResultCollection] = []
        self.current:ProjectCurrentState = ProjectCurrentState()
        self.deviation_analysis_results: List[DeviationResultCollection] = []
        self.synchronization_results: List[SynchronizationResultCollection] = []

    def to_dict(self):
        # Ergebnis-Speicher normalisieren (id->dict) + Order-Listen für History
        pre_by_id = {str(p.result_id): p.to_dict() for p in self.preprocessing_results}
        dev_by_id = {str(r.result_id): r.to_dict() for r in self.deviation_analysis_results}
        sync_by_id = {str(r.result_id): r.to_dict() for r in self.synchronization_results}

        current = {}
        if self.current is not None:
            current = {
                "preprocessing_id": str(self.current.preprocessing.result_id) if self.current.preprocessing else None,
                "deviation_id": str(self.current.deviation_analysis.result_id) if self.current.deviation_analysis else None,
                "synchronization_id": str(self.current.synchronization.result_id) if self.current.synchronization else None,
            }

        return {
            "schema_version": 2,
            "project_id": str(self.project_id),
            "title": self.title,
            "description": self.description,
            "modified_at": datetime.now(timezone.utc).isoformat(),

            "transcript": self.transcript.to_dict() if self.transcript else None,
            "asr_extract": self.asr_extract.to_dict() if self.asr_extract else None,

            "preprocessing_results_by_id": pre_by_id,
            "deviation_results_by_id": dev_by_id,
            "synchronization_results_by_id": sync_by_id,
            "run_order": {
                "preprocessing": [str(p.result_id) for p in self.preprocessing_results],
                "deviation": [str(r.result_id) for r in self.deviation_analysis_results],
                "synchronization": [str(r.result_id) for r in self.synchronization_results],
            },
            "current": current,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        obj = cls(
            title=d.get("title", ""),
            description=d.get("description", ""),
            transcript=ManualTranscript.from_dict(d["transcript"])
            if d.get("transcript") else None,
            asr_extract=ASRExtract.from_dict(d["asr_extract"])
            if d.get("asr_extract") else None,
        )

        obj.project_id = uuid.UUID(d["project_id"])
        schema = int(d.get("schema_version", 1))

        # --------- Load preprocessing ----------
        pre_map = {}
        if schema >= 2 and "preprocessing_results_by_id" in d:
            for pid, pd in d.get("preprocessing_results_by_id", {}).items():
                p = PreprocessingResultCollection.from_dict(pd)
                pre_map[str(p.result_id)] = p
            order = d.get("run_order", {}).get("preprocessing", list(pre_map.keys()))
            obj.preprocessing_results = [pre_map[x] for x in order if x in pre_map]
        else:
            # backward compat: list
            obj.preprocessing_results = [PreprocessingResultCollection.from_dict(p) for p in d.get("preprocessing_results", [])]
            pre_map = {str(p.result_id): p for p in obj.preprocessing_results}

        # --------- Load deviation ----------
        print("Project from dict: Deviation")
        dev_map = {}
        if schema >= 2 and "deviation_results_by_id" in d:
            for did, dd in d.get("deviation_results_by_id", {}).items():
                r = DeviationResultCollection.from_dict(dd)
                dev_map[str(r.result_id)] = r
            order = d.get("run_order", {}).get("deviation", list(dev_map.keys()))
            obj.deviation_analysis_results = [dev_map[x] for x in order if x in dev_map]
        else:
            obj.deviation_analysis_results = []

        # --------- Load synchronization ----------
        sync_map = {}
        if schema >= 2 and "synchronization_results_by_id" in d:
            for sid, sd in d.get("synchronization_results_by_id", {}).items():
                r = SynchronizationResultCollection.from_dict(sd)
                sync_map[str(r.result_id)] = r
            order = d.get("run_order", {}).get("synchronization", list(sync_map.keys()))
            obj.synchronization_results = [sync_map[x] for x in order if x in sync_map]
        else:
            obj.synchronization_results = []

        # --------- Resolve references / fill cache fields for GUI ----------
        # Deviation: attach transcript/extract from referenced preprocessing
        for dev in obj.deviation_analysis_results:
            pid = str(dev.preprocessing_id) if dev.preprocessing_id else None
            if pid and pid in pre_map:
                pre = pre_map[pid]
                dev.transcript_preprocessed = pre.transcript_results
                dev.extract_preprocessed = pre.extract_results

        # Sync: attach matrix + transcript/extract via referenced deviation/preprocessing
        dev_map2 = {str(r.result_id): r for r in obj.deviation_analysis_results}
        for syn in obj.synchronization_results:
            did = str(syn.deviation_id) if syn.deviation_id else None
            if did and did in dev_map2:
                dev = dev_map2[did]
                syn.similarity_matrix = dev.result_matrix
                # transcript/extract come from preprocessing referenced by deviation
                syn.transcript_preprocessed = dev.transcript_preprocessed
                syn.extract_preprocessed = dev.extract_preprocessed

        # --------- Restore current selection ----------
        cur = d.get("current", {}) or {}
        cur_pre = cur.get("preprocessing_id")
        cur_dev = cur.get("deviation_id")
        cur_syn = cur.get("synchronization_id")

        obj.current = ProjectCurrentState()
        if cur_pre and cur_pre in pre_map:
            obj.current.preprocessing = pre_map[cur_pre]
        elif obj.preprocessing_results:
            obj.current.preprocessing = obj.preprocessing_results[-1]

        if cur_dev and cur_dev in dev_map:
            obj.current.deviation_analysis = dev_map[cur_dev]
        elif obj.deviation_analysis_results:
            obj.current.deviation_analysis = obj.deviation_analysis_results[-1]

        if cur_syn and cur_syn in sync_map:
            obj.current.synchronization = sync_map[cur_syn]
        elif obj.synchronization_results:
            obj.current.synchronization = obj.synchronization_results[-1]

        return obj
    
    def has_analysis_results(self) -> bool:
        if self.preprocessing_results or self.deviation_analysis_results or self.synchronization_results:
            return True
        cur = getattr(self, "current", None)
        if cur is not None:
            if getattr(cur, "preprocessing", None) or getattr(cur, "deviation_analysis", None) or getattr(cur, "synchronization", None):
                return True
        return False
    
    def clear_analysis_results(self) -> None:
        self.preprocessing_results = []
        self.deviation_analysis_results = []
        self.synchronization_results = []
        self.current = ProjectCurrentState()

class Export:
    pass

class ExportConfig:
    pass