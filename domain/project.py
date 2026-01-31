import uuid

from datetime import datetime, timezone
from typing import List

from domain.preprocessing import PreprocessingConfig, PreprocessingResult
from domain.source_document import ManualTranscript, ASRExtract

class PreprocessingResultCollection:
    def __init__(self, config = PreprocessingConfig()):
        self.time = datetime.now()
        self.transcript_results: List[PreprocessingResult] = []
        self.extract_results: List[PreprocessingResult] = []
        self.config = config

    def to_dict(self):
        print("Type:", type(self.config))
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

class Project:
    def __init__(self, title: str = "", description: str = "", transcript: ManualTranscript = None, asr_extract: ASRExtract = None):
        self.project_id = uuid.uuid4()
        self.title = title
        self.description = description
        self.transcript = transcript
        self.asr_extract = asr_extract
        self.preprocessing_results: List[PreprocessingResultCollection] = []
        # self.deviation_analysis_results = []
        # self.alignment_results = []

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