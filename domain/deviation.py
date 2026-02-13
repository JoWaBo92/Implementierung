from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.functional import cosine_similarity

import spacy
from spacy.tokens import Doc

from typing import Iterable

from sentence_transformers import SentenceTransformer, util

from domain.source_document import ASRExtract, ManualTranscript
from domain.preprocessing import PreprocessingPipeline, PreprocessingConfig


class DeviationMethod(Enum):
    STANDARD = "Standard"
    TRF = "Transformer"
    SENT_TRF = "Sentence_Transformer"


class DeviationAnalysisConfig:
    def __init__(
        self,
        library: str = "de_core_news_md",
        method: DeviationMethod = DeviationMethod.STANDARD,
        similar_length: bool = False,
        length_min_ratio: float = 0.20,
        length_alpha: float = 0.75,
        ultra_short_tokens: int = 2,
        similar_position: bool = False,
        position_gamma: float = 4.0,
    ):
        self.library = library
        self.method = method
        self.similar_length = similar_length
        self.length_min_ratio = length_min_ratio
        self.length_alpha = length_alpha
        self.ultra_short_tokens = ultra_short_tokens
        self.similar_position = similar_position
        self.position_gamma = position_gamma

    def to_dict(self) -> dict:
        return {
            "library": self.library,
            "method": getattr(self.method, "name", str(self.method)),
            "method_value": getattr(self.method, "value", str(self.method)),
            "similar_length": bool(self.similar_length),
            "length_min_ratio": float(self.length_min_ratio),
            "length_alpha": float(self.length_alpha),
            "ultra_short_tokens": int(self.ultra_short_tokens),
            "similar_position": bool(self.similar_position),
            "position_gamma": float(self.position_gamma),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeviationAnalysisConfig":
        if not isinstance(d, dict):
            return cls()

        m = d.get("method", None)
        mv = d.get("method_value", None)
        method = DeviationMethod.STANDARD
        if isinstance(m, str):
            if m in DeviationMethod.__members__:
                method = DeviationMethod[m]
            else:
                # fallback: value-lookup
                for e in DeviationMethod:
                    if e.value == m:
                        method = e
                        break
        elif isinstance(mv, str):
            for e in DeviationMethod:
                if e.value == mv:
                    method = e
                    break
        return cls(
            library=d.get("library", "de_core_news_md"),
            method=method,
            similar_length=bool(d.get("similar_length", False)),
            length_min_ratio=float(d.get("length_min_ratio", 0.20)),
            length_alpha=float(d.get("length_alpha", 0.75)),
            ultra_short_tokens=int(d.get("ultra_short_tokens", 2)),
            similar_position=bool(d.get("similar_position", False)),
            position_gamma=float(d.get("position_gamma", 4.0)),
        )


class DeviationCalculator:
    def __init__(
        self,
        config: DeviationAnalysisConfig,
        backend: Optional[Union["spacy.language.Language", SentenceTransformer]] = None,
        device: Optional[str] = None,
    ):
        self.config = config
        self.backend = backend or self._load_backend(config, device=device)

    @staticmethod
    def _load_backend(config: DeviationAnalysisConfig, device: Optional[str] = None):
        if config.method in (DeviationMethod.STANDARD, DeviationMethod.TRF):
            return spacy.load(config.library)
        if config.method == DeviationMethod.SENT_TRF:
            if device is not None:
                return SentenceTransformer(config.library, device=device)
            return SentenceTransformer(config.library)
        raise ValueError(f"Unknown method: {config.method}")
    
    @staticmethod
    def _relative_positions(n: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=np.float32)
        if n == 1:
            return np.array([0.5], dtype=np.float32)
        idx = np.arange(n, dtype=np.float32)
        return (idx + 0.5) / float(n)
    
    @classmethod
    def _position_weight_matrix(cls, n_a: int, n_b: int, gamma: float) -> np.ndarray:
        pos_a = cls._relative_positions(n_a)[:, None]   # (n_a, 1)
        pos_b = cls._relative_positions(n_b)[None, :]   # (1, n_b)

        delta = np.abs(pos_a - pos_b)                   # (n_a, n_b)
        w = np.exp(-gamma * delta).astype(np.float32)
        return w
    
    @staticmethod
    def _doc_length(x: Any) -> int:
        if isinstance(x, Doc):
            return max(0, len(x))
        if isinstance(x, str):
            return len(x.split())

        doc = getattr(x, "doc", None)
        if isinstance(doc, Doc):
            return max(0, len(doc))

        txt = getattr(x, "clean_text", None)
        if isinstance(txt, str):
            return len(txt.split())

        txt = getattr(x, "raw_text", None)
        if isinstance(txt, str):
            return len(txt.split())

        # Fallback
        return 0
    
    @classmethod
    def _length_weight_matrix(cls, items_a: Sequence[Any], items_b: Sequence[Any], min_ratio: float, alpha: float, ultra_short_tokens: int = 2) -> np.ndarray:
        la = np.array([cls._doc_length(x) for x in items_a], dtype=np.float32)  
        lb = np.array([cls._doc_length(x) for x in items_b], dtype=np.float32) 

        A = la[:, None]
        B = lb[None, :]

        mx = np.maximum(A, B)
        mn = np.minimum(A, B)

        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(mx > 0, mn / mx, 0.0).astype(np.float32)

        w = np.where(r >= float(min_ratio), np.power(r, float(alpha)), 0.0).astype(np.float32)

        if ultra_short_tokens is not None and int(ultra_short_tokens) > 0:
            w = np.where(mn < float(ultra_short_tokens), 0.0, w).astype(np.float32)

        return w
    

    # ---------- Calculate whole similarity matrix ----------

    def similarity_matrix(self, items_a: Sequence[Any], items_b: Sequence[Any]) -> np.ndarray:
        if self.config.method == DeviationMethod.SENT_TRF:
            model = self._require_sentence_transformer()

            texts_a = [self._as_text(x) for x in items_a]
            texts_b = [self._as_text(x) for x in items_b]

            emb_a = model.encode(texts_a, convert_to_tensor=True, normalize_embeddings=True)
            emb_b = model.encode(texts_b, convert_to_tensor=True, normalize_embeddings=True)

            sim = util.cos_sim(emb_a, emb_b).cpu().numpy().astype(np.float32)

        elif self.config.method == DeviationMethod.STANDARD:
            docs_a = [self._as_doc(x) for x in items_a]
            docs_b = [self._as_doc(x) for x in items_b]

            A = np.vstack([d.vector for d in docs_a]).astype(np.float32)
            B = np.vstack([d.vector for d in docs_b]).astype(np.float32)

            A = self._l2_normalize_np(A)
            B = self._l2_normalize_np(B)

            sim = (A @ B.T).astype(np.float32)

        elif self.config.method == DeviationMethod.TRF:
            docs_a = [self._as_doc(x) for x in items_a]
            docs_b = [self._as_doc(x) for x in items_b]

            EA = self._docs_trf_embeddings_mean(docs_a)  # Tensor (len(a), dim)
            EB = self._docs_trf_embeddings_mean(docs_b)  # Tensor (len(b), dim)

            EA = torch.nn.functional.normalize(EA, p=2, dim=1)
            EB = torch.nn.functional.normalize(EB, p=2, dim=1)

            sim = (EA @ EB.T).cpu().numpy().astype(np.float32)

        else:
            raise ValueError(f"Unknown method: {self.config.method}")

        if getattr(self.config, "similar_length", False) and sim.size > 0:
            w_len = self._length_weight_matrix(
                items_a, items_b,
                min_ratio=float(getattr(self.config, "length_min_ratio", 0.20)),
                alpha=float(getattr(self.config, "length_alpha", 0.75)),
                ultra_short_tokens=int(getattr(self.config, "ultra_short_tokens", 2))
            )
            sim = sim * w_len

        if getattr(self.config, "similar_position", False) and sim.size > 0:
            w_pos = self._position_weight_matrix(sim.shape[0], sim.shape[1], gamma=float(self.config.position_gamma))
            sim = sim * w_pos

        return sim


    # ---------- Calculate similarity between single values ----------

    def similarity(self, a: Union[str, Doc], b: Union[str, Doc]) -> float:
        if self.config.method == DeviationMethod.SENT_TRF:
            return float(self._sim_sentence_transformer(a, b))
        elif self.config.method == DeviationMethod.TRF:
            return float(self._sim_spacy_trf(a, b))
        else:  # STANDARD
            return float(self._sim_spacy_standard(a, b))

    def _sim_spacy_standard(self, a: Union[str, Doc], b: Union[str, Doc]) -> float:
        nlp = self._require_spacy()
        doc1 = a if isinstance(a, Doc) else nlp(a)
        doc2 = b if isinstance(b, Doc) else nlp(b)
        return doc1.similarity(doc2)

    def _sim_spacy_trf(self, a: Union[str, Doc], b: Union[str, Doc]) -> float:
        nlp = self._require_spacy()
        doc1 = a if isinstance(a, Doc) else nlp(a)
        doc2 = b if isinstance(b, Doc) else nlp(b)

        v1 = self._doc_embedding_mean(doc1)  # Tensor (dim,)
        v2 = self._doc_embedding_mean(doc2)  # Tensor (dim,)

        return cosine_similarity(v1, v2, dim=0).item()
    
    def _sim_sentence_transformer(self, a: Union[str, Doc], b: Union[str, Doc]) -> float:
        model = self._require_sentence_transformer()

        s1 = a.text if isinstance(a, Doc) else a
        s2 = b.text if isinstance(b, Doc) else b
        if not isinstance(s1, str) or not isinstance(s2, str):
            raise TypeError("SENT_TRF expects strings or spaCy Docs (Doc.text).")

        emb = model.encode([s1, s2], convert_to_tensor=True)  # Tensor (2, dim)
        return util.cos_sim(emb[0], emb[1]).item()
    

    # ---------- Helper functions ----------

    def _doc_embedding_mean(self, doc: Doc) -> torch.Tensor:
        if not hasattr(doc._, "trf_data") or doc._.trf_data is None:
            raise ValueError("Doc has no trf_data.")

        ragged = doc._.trf_data.last_hidden_layer_state  # thinc.types.Ragged
        data = ragged.data
        lengths = ragged.lengths

        # doc-by-doc
        seq_len = int(lengths[0])
        seq = data[:seq_len]

        t = torch.as_tensor(seq)  # (tokens, dim)
        return t.mean(dim=0)      # (dim,)
    
    @staticmethod
    def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom = np.maximum(denom, eps)
        return x / denom
    

    def _docs_trf_embeddings_mean(self, docs: Sequence[Doc]) -> torch.Tensor:
        embs = [self._doc_embedding_mean(d) for d in docs]  # list[Tensor (dim,)]
        return torch.stack(embs, dim=0)  # Tensor (n_docs, dim)
    

    def _as_doc(self, x: Any) -> Doc:
        if isinstance(x, Doc):
            return x
        
        doc = getattr(x, "doc", None)
        if isinstance(doc, Doc):
            return doc
        raise TypeError("Expected spaCy Doc or an object with attribute .doc (spaCy Doc).")
    

    def _as_text(self, x: Any) -> str:
        if isinstance(x, str):
            return x
        if isinstance(x, Doc):
            return x.text
        
        raw = getattr(x, "raw_text", None)
        if isinstance(raw, str):
            return raw
        clean = getattr(x, "clean_text", None)
        if isinstance(clean, str):
            return clean
        doc = getattr(x, "doc", None)
        if isinstance(doc, Doc):
            return doc.text
        raise TypeError("Expected str/Doc or an object with raw_text/clean_text/doc.")
    

    # ---------- backend helpers ----------

    def _require_spacy(self):
        if not hasattr(self.backend, "__call__"):
            raise TypeError("Backend cannot be called.")
        
        if isinstance(self.backend, SentenceTransformer):
            raise TypeError("Backend is not spaCy.")
        return self.backend


    def _require_sentence_transformer(self):
        if not isinstance(self.backend, SentenceTransformer):
            raise TypeError("Backend is not SentenceTransformer.")
        return self.backend



def plot_similarity_matrix(result_table):
    data = np.array(result_table)

    plt.figure()
    plt.imshow(data)
    plt.colorbar(label="Similarity")

    plt.xlabel("Transcript-Segmente")
    plt.ylabel("ASR-Segmente")
    plt.title("Ähnlichkeitsmatrix (Deviation Analysis)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    extract = ASRExtract("FaPra Timealignment\ADG3149_01_01_de_speaker.csv")
    transcript = ManualTranscript("FaPra Timealignment\ADG3149_01_01.odt")
    print("Source documents loaded")

    # pipe = PreprocessingPipeline()
    pipe = PreprocessingPipeline(config=PreprocessingConfig(spacy_model="de_dep_news_trf"))

    pre_extract = pipe.run_batch(extract.segments)
    pre_transcript = pipe.run_batch(transcript.segments)
    print("Source documents preprocessed")

    # calc = DeviationCalculator(DeviationAnalysisConfig("de_core_news_md", DeviationMethod.STANDARD))
    calc = DeviationCalculator(DeviationAnalysisConfig("de_dep_news_trf", DeviationMethod.TRF))
    # calc = DeviationCalculator(DeviationAnalysisConfig("paraphrase-multilingual-MiniLM-L12-v2", DeviationMethod.SENT_TRF))


    #result_table = []
    #for r, doc1 in enumerate(pre_extract):
    #    row = []
    #    for c, doc2 in enumerate(pre_transcript):
    #        # row.append(calc.similarity(doc1.doc, doc2.doc))
    #        row.append(calc.similarity(doc1.raw_text, doc2.raw_text))
    #    print(f"{r} of {len(pre_extract)} done.")
    #    result_table.append(row)

    texts_a = [r.doc for r in pre_extract]
    texts_b = [r.doc for r in pre_transcript]

    result_table = calc.similarity_matrix(texts_a, texts_b)  # numpy array

    ref_sim = calc.similarity(pre_transcript[0].doc, pre_extract[0].doc)

    plot_similarity_matrix(result_table)

    print(ref_sim, result_table[0][0])

    # 0.43596245238900666, nan, tensor([[0.0398]])