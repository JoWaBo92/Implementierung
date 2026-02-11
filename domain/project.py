import uuid

from dataclasses import dataclass

import numpy as np

from spacy.tokens import Doc

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from domain.deviation import DeviationAnalysisConfig
from domain.preprocessing import PreprocessingConfig, PreprocessingResult
from domain.source_document import ManualTranscript, ASRExtract
from domain.synchronization import SynchronizationConfig

class PreprocessingResultCollection:
    def __init__(self, config = PreprocessingConfig()):
        self.time = datetime.now()
        self.transcript_results: List[PreprocessingResult] = []
        self.extract_results: List[PreprocessingResult] = []
        self.config: PreprocessingConfig = config

    def to_dict(self):
        return {
            "time": self.time.isoformat(), 
            "transcript_results": [r.to_dict() for r in self.transcript_results],
            "extract_results": [r.to_dict() for r in self.extract_results],
            "config": self.config.to_dict()
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessingResultCollection":
        cfg = PreprocessingConfig.from_dict(d.get("config", {}))
        obj = cls(config=cfg)
        t = d.get("time")
        if t:
            obj.time = datetime.fromisoformat(t)
        obj.transcript_results = [PreprocessingResult.from_dict(x) for x in d.get("transcript_results", [])]
        obj.extract_results = [PreprocessingResult.from_dict(x) for x in d.get("extract_results", [])]
        return obj
    
class DeviationResultCollection:
    def __init__(self, config = DeviationAnalysisConfig()):
        self.time = datetime.now()
        self.config: DeviationAnalysisConfig = config
        self.transcript_preprocessed: List[PreprocessingResult] = []
        self.extract_preprocessed: List[PreprocessingResult] = []
        self.result_matrix: np.ndarray = None

@dataclass
class AlignedSegment:
    transcript_index: int
    extract_j0: int
    extract_j1: int
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    mean_similarity: Optional[float] = None

class SynchronizationResultCollection:
    def __init__(
        self,
        config: SynchronizationConfig = SynchronizationConfig(),
        transcript_preprocessed: List[PreprocessingResult] = [],
        extract_preprocessed: List[PreprocessingResult] = [],
        similarity_matrix: Optional[np.ndarray] = None,
        alignment_path: List[Tuple[int, int]] = [],
        alignment_ranges_by_transcript: List[Tuple[int, int]] = [],
        total_cost: Optional[float] = None,
        mean_similarity_on_path: Optional[float] = None,
        aligned_transcript: List[AlignedSegment] = []
    ):
        self.time = datetime.now()
        self.config = config
        self.transcript_preprocessed = transcript_preprocessed
        self.extract_preprocessed = extract_preprocessed
        self.similarity_matrix = similarity_matrix
        self.alignment_path = alignment_path
        self.alignment_ranges_by_transcript = alignment_ranges_by_transcript
        self.total_cost = total_cost
        self.mean_similarity_on_path = mean_similarity_on_path
        self.aligned_transcript: List[AlignedSegment] = aligned_transcript

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
        return {
            "project_id": str(self.project_id),
            "title": self.title,
            "description": self.description,
            "modified_at": datetime.now(timezone.utc).isoformat(),
            "preprocessing_results": [p.to_dict() for p in self.preprocessing_results],
            # "deviation_analysis_results": [r.to_dict() for r in self.deviation_analysis_results],
            # "alignment_results": [r.to_dict() for r in self.alignment_results],
            "sources": {
                "transcript": None,
                "asr_extract": None,
            }
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
        obj.preprocessing_results = [PreprocessingResultCollection.from_dict(p) for p in d.get("preprocessing_results", [])]

        # obj.deviation_analysis_results = [DeviationAnalysisResult.from_dict(r)for r in d.get("deviation_analysis_results", [])]
        # obj.alignment_results = [AlignmentResult.from_dict(r)for r in d.get("alignment_results", [])]

        return obj

class Export:
    pass

class ExportConfig:
    pass