from abc import ABC, abstractmethod
import unicodedata
import re

from domain.source_document import ASRExtract, ManualTranscript

class PreprocessingPipeline:
    pass

class PipelineStage(ABC):
    @abstractmethod
    def apply(self):
        pass

class TokenizationStage(PipelineStage):
    def apply(self):
        pass

class NormalizationStage(PipelineStage):
    def apply(self, input):

        # Normalize unicode
        s = unicodedata.normalize("NFC", input)

        # Normalize whitespaces
        s = s.replace("\u00A0", " ")
        s = re.sub(r"\s+", " ", s).strip()

        # Convert to lower case
        s = s.lower()

        return s

class PreprocessingResult:
    pass

class PreprocessingConfig:
    pass

class Token:
    text: str
    index: int

if __name__ == "__main__":
    extract = ASRExtract("FaPra Timealignment\ADG3149_01_01_de_speaker.csv")
    transcript = ManualTranscript("FaPra Timealignment\ADG3149_01_01.odt")

    norm = NormalizationStage()

    print(extract.segments[0].text)

    print(norm.apply(extract.segments[0].text))