from typing import List, Optional, Sequence
from abc import ABC, abstractmethod
import unicodedata
import re

import spacy
from spacy.language import Language

from domain.source_document import ASRExtract, ManualTranscript

class PreprocessingConfig:
    def __init__(self, spacy_model: str = "de_core_news_md", lowercase: bool = False, keep_punct: bool = True) -> None:
        self.spacy_model = spacy_model
        self.lowercase = lowercase
        self.keep_punct = keep_punct

    def to_dict(self):
        return {"spacy_model": self.spacy_model}

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessingConfig":
        return cls(spacy_model=d.get("spacy_model", "de_core_news_md"))

class Token:
    def __init__(self, t:spacy.tokens.token.Token):
        self.spacy_token: spacy.tokens.token.Token = t
        self.text: str = t.text
        self.lemma = t.lemma
        self.pos = t.pos
        self.is_stop: bool = bool(t.is_stop)
        self.is_punct: bool = bool(t.is_punct)
        self.start_char: int = int(t.idx)
        self.end_char: int = int(t.idx + len(t.text))

    def to_dict(self):
        return {
            "text": self.text,
            "lemma": self.lemma,
            "pos": self.pos,
            "is_stop": self.is_stop,
            "is_punct": self.is_punct,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Token":
        return cls(**d)

    def __str__(self):
        return (f"Token(text='{self.text}', lemma='{self.lemma}', pos={self.pos}, "
                f"is_stop={self.is_stop}, is_punct={self.is_punct}, "
                f"start_char={self.start_char}, end_char={self.end_char})")

class PreprocessingResult:
    def __init__(self):
        self.raw_text: str = ""
        self.clean_text: str = ""
        self.tokens: List[Token] = []
        self.config = None
        self.doc = None

    def to_dict(self):
        return {
            "raw_text": self.raw_text,
            "clean_text": self.clean_text,
            "tokens": [t.to_dict() for t in self.tokens],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessingResult":
        obj = cls()
        obj.raw_text = d.get("raw_text", "")
        obj.clean_text = d.get("clean_text", "")
        obj.tokens = [Token.from_dict(td) for td in d.get("tokens", [])]
        return obj

class PipelineStage(ABC):
    @abstractmethod
    def apply(self, result, config, nlp):
        pass


class NormalizationStage(PipelineStage):
    def apply(self, result: PreprocessingResult, config: PreprocessingConfig, nlp):

        text = result.raw_text

        # Normalize unicode
        s = unicodedata.normalize("NFC", text)

        # Normalize whitespaces
        s = s.replace("\u00A0", " ")
        s = re.sub(r"\s+", " ", s).strip()

        # Convert to lower case
        if config.lowercase:
            s = s.lower()

        result.clean_text = s

        return result

class TokenizationStage(PipelineStage):
    def apply(self, result: PreprocessingResult, config: PreprocessingConfig, nlp):
        result.tokens = []
        doc = nlp(result.clean_text)
        result.doc = doc
        
        for t in doc:
            if t.is_space:
                continue
            if (not config.keep_punct) and t.is_punct:
                continue
            token = Token(t)
            result.tokens.append(token)

        return result

class PreprocessingPipeline:
    def __init__(self,config: Optional[PreprocessingConfig] = None,stages: Optional[Sequence[PipelineStage]] = None):
        self.config = config or PreprocessingConfig()
        self.stages = list(stages) if stages is not None else [NormalizationStage(), TokenizationStage()]
        self.nlp = spacy.load(self.config.spacy_model)

    def run(self, segment) -> PreprocessingResult:
        result = PreprocessingResult()
        result.raw_text = segment.text
        for stage in self.stages:
            result = stage.apply(result=result, config=self.config, nlp=self.nlp)
        return result
    
    def run_batch(self, segments) -> List[PreprocessingResult]:
        results = []
        for s in segments:
            results.append(self.run(s))
        return results


if __name__ == "__main__":
    extract = ASRExtract("FaPra Timealignment\ADG3149_01_01_de_speaker.csv")
    transcript = ManualTranscript("FaPra Timealignment\ADG3149_01_01.odt")

    nlp = spacy.load("de_core_news_md")
    norm = NormalizationStage()
    tk = TokenizationStage()

    print(extract.segments[0].text)

    print("Test")
    pipe = PreprocessingPipeline(stages=[norm, tk])
    test_res = pipe.run_batch(extract.segments)
