import torch
from torch.nn.functional import cosine_similarity

from sentence_transformers import SentenceTransformer, util

import spacy
import time

from enum import Enum

from domain.source_document import ASRExtract, ManualTranscript
from domain.preprocessing import PreprocessingPipeline, Token
from domain.project import PreprocessingResultCollection

class AlignMethod(Enum):
    STANDARD = "Standard"
    TRF = "Transformer"
    SENT_TRF = "Sentence_Transformer"

class AlignmentResult:
    def __init__(self, method: AlignMethod):
        if method == AlignMethod.STANDARD:
            pass

class AlignmentConfig:
    pass

class ManualCorrection:
    pass

class Match:
    pass

def doc_embedding_mean(doc) -> torch.Tensor:
    ragged = doc._.trf_data.last_hidden_layer_state  # thinc.types.Ragged
    data = ragged.data  # shape: (sum_tokens_over_docs, hidden_dim)
    lengths = ragged.lengths  # shape: (n_docs_in_batch,)

    # Falls du doc-by-doc nlp(...) machst, ist hier i.d.R. nur 1 Sequenz drin:
    if len(lengths) == 1:
        seq = data[: lengths[0]]
    else:
        # sicherheitshalber: erste Sequenz nehmen
        start = 0
        end = int(lengths[0])
        seq = data[start:end]

    # Falls data ein cupy-array wäre (GPU), erst auf CPU holen:
    if hasattr(seq, "get"):
        seq = seq.get()

    t = torch.as_tensor(seq)           # (tokens, dim)
    return t.mean(dim=0)               # (dim,)

if __name__ == "__main__":
    print(AlignMethod.TRF.value)

    extract = ASRExtract("FaPra Timealignment\ADG3149_01_01_de_speaker.csv")
    transcript = ManualTranscript("FaPra Timealignment\ADG3149_01_01.odt")

    # pipe = PreprocessingPipeline()
    # pre_result_transcript = pipe.run_batch(transcript.segments)
    # pre_result_extract = pipe.run_batch(extract.segments)

    # result_collection = PreprocessingResultCollection()
    # result_collection.config = pipe.config
    # result_collection.transcript_results = pre_result_transcript
    # result_collection.extract_results = pre_result_extract

    nlp = spacy.load("de_core_news_md")
    nlp_t = spacy.load("de_dep_news_trf")
    transcript_seg = transcript.segments[1].text

    transcript_nlp_t = nlp_t(transcript_seg)
    transcript_nlp = nlp(transcript_seg)
    # transcript_seg = nlp("Wie und wo bist du groß geworden?")

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    for i in range(20):
        extract_seg = extract.segments[i].text
        es_nlp_t = nlp_t(extract_seg)
        es_nlp = nlp(extract_seg)

        v1 = doc_embedding_mean(transcript_nlp_t)
        v2 = doc_embedding_mean(es_nlp_t)

        sim = transcript_nlp.similarity(es_nlp)
        sim_t = cosine_similarity(v1, v2, dim=0).item()

        emb = model.encode([transcript_seg, extract_seg])
        sim2 = util.cos_sim(emb[0], emb[1])

        print(f"'{transcript_seg}' vs. \n'{extract_seg}': \n{sim}, {sim_t}, {sim2}")
        print("")

    print("Done!")
