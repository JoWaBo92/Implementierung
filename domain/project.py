import uuid
from domain.source_document import ManualTranscript, ASRExtract

class Project:
    def __init__(self, title: str = "", description: str = "", transcript: ManualTranscript = None, asr_extract: ASRExtract = None):
        self.project_id = uuid.uuid4()
        self.title = title
        self.description = description
        self.transcript = transcript
        self.asr_extract = asr_extract
        self.preprocessing_pipeline = None
        self.preprocessing_results = []
        self.deviation_analysis_results = []
        self.alignment_results = []

class Export:
    pass

class ExportConfig:
    pass