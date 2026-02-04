import torch
from torch.nn.functional import cosine_similarity

from sentence_transformers import SentenceTransformer, util

import spacy
import time

from enum import Enum

from domain.source_document import ASRExtract, ManualTranscript
from domain.preprocessing import PreprocessingPipeline, Token, PreprocessingConfig
from domain.project import PreprocessingResultCollection

class DeviationMethod(Enum):
    STANDARD = "Standard"
    TRF = "Transformer"
    SENT_TRF = "Sentence_Transformer"

class DeviationAnalysisResult:
    pass

class DeviationAnalysisConfig:
    def __init__(self, library: str, method: DeviationMethod = DeviationMethod.STANDARD):
        self.method = method
        self.library = library

class Deviation:
    def __init__(self, config: DeviationAnalysisConfig, doc1, doc2):
        self.doc1 = doc1
        self.doc2 = doc2
        self.config = config

        if config.method == DeviationMethod.STANDARD or config.method == DeviationMethod.STANDARD:
            self.nlp = spacy.load(config.library)
        else:
            self.nlp = SentenceTransformer(config.library)

        self.deviation = self.calculate()

    def calculate(self):
        print(f"{self.config.method}, {type(self.doc1)}, {type(self.doc2)}: {self.config.method == DeviationMethod.STANDARD}")
        if self.config.method == DeviationMethod.STANDARD and type(self.doc1) is spacy.tokens.doc.Doc and type(self.doc2) is spacy.tokens.doc.Doc:
            return self._calculate_standard(self.doc1, self.doc2)
        elif self.config.method == DeviationMethod.TRF and type(self.doc1) is spacy.tokens.doc.Doc and type(self.doc2) is spacy.tokens.doc.Doc:
            return self._calculate_transformer(self.doc1, self.doc2)
        elif self.config.method == DeviationMethod.SENT_TRF and type(self.doc1) is str and type(self.doc2) is str:
            return self._calculate_sentence_transformer(self.doc1, self.doc2)
        else:
            return None

    def _calculate_standard(self, doc1: spacy.tokens.doc.Doc, doc2: spacy.tokens.doc.Doc):
        return doc1.similarity(doc2)

    def _calculate_transformer(self, doc1: spacy.tokens.doc.Doc, doc2: spacy.tokens.doc.Doc):
        v1 = self._doc_embedding_mean(doc1)
        v2 = self._doc_embedding_mean(doc2)

        return cosine_similarity(v1, v2, dim=0).item()

    def _calculate_sentence_transformer(self, doc1: str, doc2: str):
        emb = self.nlp.encode([doc1, doc2])
        return util.cos_sim(emb[0], emb[1])

    def _doc_embedding_mean(self, doc) -> torch.Tensor:
        ragged = doc._.trf_data.last_hidden_layer_state 
        data = ragged.data  
        lengths = ragged.lengths 

        start = 0
        end = int(lengths[0])
        seq = data[start:end]

        t = torch.as_tensor(seq)         
        return t.mean(dim=0)          


if __name__ == "__main__":
    extract = ASRExtract("FaPra Timealignment\ADG3149_01_01_de_speaker.csv")
    transcript = ManualTranscript("FaPra Timealignment\ADG3149_01_01.odt")

    # pipe = PreprocessingPipeline()

    # pre_extract = pipe.run_batch(extract.segments)
    # pre_transcript = pipe.run_batch(transcript.segments)

    # standard_config = DeviationAnalysisConfig("de_core_news_md")
    # dev_standard = Deviation(standard_config, pre_extract[0].doc, pre_transcript[1].doc)
    # print(dev_standard.deviation)

    # pipe_trf = PreprocessingPipeline(config=PreprocessingConfig(spacy_model="de_dep_news_trf"))

    # pre_extract_trf = pipe_trf.run(extract.segments[0])
    # pre_transcript_trf = pipe_trf.run(transcript.segments[0])

    # trf_config = DeviationAnalysisConfig("de_dep_news_trf", DeviationMethod.TRF)
    # dev_trf = Deviation(trf_config, pre_extract_trf.doc, pre_transcript_trf.doc)
    # print(dev_trf.deviation)

    sent_trf_config = DeviationAnalysisConfig("paraphrase-multilingual-MiniLM-L12-v2", DeviationMethod.SENT_TRF)
    dev_sent_trf = Deviation(sent_trf_config, extract.segments[0].text, transcript.segments[1].text)
    print(dev_sent_trf.deviation)

    # 0.43596245238900666, nan, tensor([[0.0398]])