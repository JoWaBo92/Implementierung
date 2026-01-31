from typing import List
from abc import ABC, abstractmethod
import unicodedata
import re

import spacy
from spacy.language import Language

from domain.source_document import ASRExtract, ManualTranscript

class Token:
    def __init__(self, text: str, lemma: str, pos: str,
                 is_stop: bool, is_punct: bool,
                 start_char: int, end_char: int):
        self.text = text
        self.lemma = lemma
        self.pos = pos
        self.is_stop = is_stop
        self.is_punct = is_punct
        self.start_char = start_char
        self.end_char = end_char

    def __str__(self):
        return (f"Token(text='{self.text}', lemma='{self.lemma}', pos={self.pos}, "
                f"is_stop={self.is_stop}, is_punct={self.is_punct}, "
                f"start_char={self.start_char}, end_char={self.end_char})")

class PreprocessingResult:
    def __init__(self):
        self.raw_text: str = ""
        self.clean_text: str = ""
        self.tokens: List[Token] = []

class PipelineStage(ABC):
    @abstractmethod
    def apply(self, result, nlp):
        pass


class NormalizationStage(PipelineStage):
    def apply(self, result: PreprocessingResult, nlp):

        text = result.raw_text

        # Normalize unicode
        s = unicodedata.normalize("NFC", text)

        # Normalize whitespaces
        s = s.replace("\u00A0", " ")
        s = re.sub(r"\s+", " ", s).strip()

        # Convert to lower case
        s = s.lower()

        result.clean_text = s

        return result

class TokenizationStage(PipelineStage):
    def apply(self, result: PreprocessingResult, nlp):
        doc = nlp(result.clean_text)
        
        for t in doc:
            if t.is_space:
                continue
            if t.is_punct:
                continue
            token = Token(text=t.text,lemma=t.lemma_,pos=t.pos_,is_stop=bool(t.is_stop),
                    is_punct=bool(t.is_punct),start_char=int(t.idx),end_char=int(t.idx + len(t.text)))
            result.tokens.append(token)

        return result

class PreprocessingConfig:
    def __init__(self, spacy_model: str = "de_core_news_md") -> None:
        self.spacy_model = spacy_model

class PreprocessingPipeline:
    def __init__(self, config: PreprocessingConfig = PreprocessingConfig(), stages: List[PipelineStage] = [NormalizationStage(), TokenizationStage()]):
        self.config = config
        self.stages = stages
        self.nlp = spacy.load(config.spacy_model)

    def run(self, segment):
        result = PreprocessingResult()
        result.raw_text = segment.text
        for stage in self.stages:
            result = stage.apply(result, self.nlp)
        return result
    
    def run_batch(self, segments):
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
