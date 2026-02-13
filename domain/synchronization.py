import torch
from torch.nn.functional import cosine_similarity

import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util

import spacy
import time

from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple
from dataclasses import dataclass

from domain.source_document import ASRExtract, ManualTranscript
from domain.preprocessing import PreprocessingPipeline, PreprocessingConfig, Token
from domain.deviation import DeviationAnalysisConfig, DeviationCalculator, DeviationMethod

class AlignmentAlgorithm(Enum):
    DTW = "DTW"

@dataclass(frozen=True)
class SynchronizationConfig:
    algorithm: AlignmentAlgorithm = AlignmentAlgorithm.DTW

    step_v: float = 0.10
    step_h: float = 0.12

    min_sim: Optional[float] = 0.15
    big_cost: float = 1e6

    band: Optional[int] = None

@dataclass
class SynchronizationResult:
    path: List[Tuple[int, int]]                    
    total_cost: float
    mean_similarity_on_path: float
    ranges_by_transcript: List[Tuple[int, int]]    

class SynchronizationCalculator:
    def __init__(self, config: SynchronizationConfig):
        self.config = config

    def synchronize(self, sim_matrix: np.ndarray) -> SynchronizationResult:
        if self.config.algorithm == AlignmentAlgorithm.DTW:
            return self._dtw_align(sim_matrix)
        raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
    
    def _dtw_align(self, sim: np.ndarray) -> SynchronizationResult:
        if sim.ndim != 2:
            raise ValueError("sim_matrix must be 2D (n_transcript x n_extract)")
        nT, nE = sim.shape
        if nT == 0 or nE == 0:
            return SynchronizationResult(
                path=[],
                total_cost=float("inf"),
                mean_similarity_on_path=float("nan"),
                ranges_by_transcript=[(-1, -1)] * nT,
            )

        step_v = float(self.config.step_v)
        step_h = float(self.config.step_h)
        min_sim = self.config.min_sim
        big_cost = float(self.config.big_cost)
        band = self.config.band

        # Convert similarity -> cost
        cost = (1.0 - sim).astype(np.float32)
        if min_sim is not None:
            cost = np.where(sim < float(min_sim), big_cost, cost).astype(np.float32)

        # DP + backpointers
        D = np.full((nT, nE), np.float32(big_cost), dtype=np.float32)
        P = np.full((nT, nE), np.int8(-1), dtype=np.int8)  # 0=diag, 1=up(i-1,j), 2=left(i,j-1)

        def _j_range(i: int) -> Tuple[int, int]:
            if band is None:
                return 0, nE
            j0 = max(0, i - band)
            j1 = min(nE, i + band + 1)
            return j0, j1

        for i in range(nT):
            j0, j1 = _j_range(i)
            for j in range(j0, j1):
                c = cost[i, j]

                if i == 0 and j == 0:
                    D[i, j] = c
                    P[i, j] = 0
                    continue

                best = big_cost
                best_move = -1

                # diagonal (match)
                if i > 0 and j > 0:
                    v = float(D[i - 1, j - 1])
                    if v < best:
                        best = v
                        best_move = 0

                # up (consume transcript) => allows n:1
                if i > 0:
                    v = float(D[i - 1, j]) + step_h
                    if v < best:
                        best = v
                        best_move = 1

                # left (consume extract) => allows 1:n
                if j > 0:
                    v = float(D[i, j - 1]) + step_v
                    if v < best:
                        best = v
                        best_move = 2

                D[i, j] = np.float32(c + best)
                P[i, j] = np.int8(best_move)

        # Backtrack
        i, j = nT - 1, nE - 1
        path: List[Tuple[int, int]] = [(i, j)]
        while not (i == 0 and j == 0):
            move = int(P[i, j])
            if move == 0:
                i -= 1
                j -= 1
            elif move == 1:
                i -= 1
            elif move == 2:
                j -= 1
            else:
                # Unreachable state can happen with tight band + min_sim gating
                break
            path.append((i, j))
        path.reverse()

        # Mean similarity along path
        sims_on_path = [float(sim[ii, jj]) for ii, jj in path] if path else []
        mean_sim = float(np.mean(sims_on_path)) if sims_on_path else float("nan")

        ranges = self._ranges_by_transcript(path, nT)

        return SynchronizationResult(
            path=path,
            total_cost=float(D[nT - 1, nE - 1]),
            mean_similarity_on_path=mean_sim,
            ranges_by_transcript=ranges,
        )
    
    @staticmethod
    def _ranges_by_transcript(path: Sequence[Tuple[int, int]], nT: int) -> List[Tuple[int, int]]:
        mins: List[Optional[int]] = [None] * nT
        maxs: List[Optional[int]] = [None] * nT

        for i, j in path:
            if 0 <= i < nT:
                if mins[i] is None or j < mins[i]:
                    mins[i] = j
                if maxs[i] is None or j > maxs[i]:
                    maxs[i] = j

        out: List[Tuple[int, int]] = []
        for i in range(nT):
            if mins[i] is None:
                out.append((-1, -1))
            else:
                out.append((int(mins[i]), int(maxs[i])))
        return out


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

def plot_alignment(sim_matrix: np.ndarray, path: list[tuple[int, int]], title: str = "DTW-Alignment auf Ähnlichkeitsmatrix"):
    is_ = [i for i, j in path]
    js_ = [j for i, j in path]

    plt.figure()
    plt.imshow(sim_matrix, aspect="auto", origin="lower")
    plt.colorbar(label="Similarity")

    plt.plot(js_, is_, linewidth=1.5)  

    plt.xlabel("ASR-Extrakt-Segmente (j)")
    plt.ylabel("Transkript-Segmente (i)")
    plt.title(title)

    plt.tight_layout()
    plt.show()

def path_moves(path):
    moves = {}
    for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
        if i1 == i0 + 1 and j1 == j0 + 1:
            moves[(i1, j1)] = "diag"
        elif i1 == i0 + 1 and j1 == j0:
            moves[(i1, j1)] = "up"      # n:1 (transcript advances)
        elif i1 == i0 and j1 == j0 + 1:
            moves[(i1, j1)] = "left"    # 1:n (extract advances)
    return moves

if __name__ == "__main__":
    extract = ASRExtract("FaPra Timealignment\ADG3149_01_01_de_speaker.csv")
    transcript = ManualTranscript("FaPra Timealignment\ADG3149_01_01.odt")
    print("Source documents loaded")

    pipe = PreprocessingPipeline(config=PreprocessingConfig(spacy_model="de_dep_news_trf"))
    extract_results = pipe.run_batch(extract.segments)
    transcript_results = pipe.run_batch(transcript.segments)
    print("Source documents preprocessed")

    deviation_config = DeviationAnalysisConfig(method=DeviationMethod.SENT_TRF, library="paraphrase-multilingual-MiniLM-L12-v2", similar_length=False, similar_position=True)
    deviation_calc = DeviationCalculator(config=deviation_config)

    transcript_docs = [r.doc for r in transcript_results]
    extract_docs = [r.doc for r in extract_results]

    sim_matrix = deviation_calc.similarity_matrix(transcript_docs, extract_docs)

    plot_similarity_matrix(sim_matrix)

    synch_config = SynchronizationConfig()
    synch_calc = SynchronizationCalculator(config=synch_config)

    synch_results = synch_calc.synchronize(sim_matrix=sim_matrix)
    print(f"Total cost: {synch_results.total_cost}")

    moves = path_moves(synch_results.path)
    for i, (j0, j1) in enumerate(synch_results.ranges_by_transcript):
        tr_text = transcript_results[i].clean_text

        print("=" * 80)
        print(f"[T {i:03d}] {tr_text}")

        if j0 < 0:
            print("  -> no aligned ASR segments")
            continue

        print(f"  -> ASR segments [{j0}..{j1}]")

        for j in range(j0, j1 + 1):
            asr_text = extract_results[j].clean_text
            sim = sim_matrix[i, j]

            move = moves.get((i, j), "start")

            print(f"     [E {j:03d}] sim={sim:0.3f}  path={move}")
            print(f"           {asr_text}")
    
    plot_alignment(sim_matrix, synch_results.path)