import uuid

class project:
    def __init__(self):
        self.project_id = uuid.uuid4()
        self.title = ""
        self.description = ""
        self.transcript = None
        self.asr_extract = None
        self.preprocessing_pipeline = None
        self.preprocessing_results = []
        self.deviation_analysis_results = []
        self.alignment_results = []

class Export:
    pass

class ExportConfig:
    pass