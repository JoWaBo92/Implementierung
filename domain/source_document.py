import uuid
from odf.opendocument import load
from odf import text as odf_text
from odf import teletype

from typing import List, Optional, Dict, Any, Union, Tuple

class TranscriptSegment:
    def __init__(self, text: str, speaker: str, index: int, segment_id: Optional[uuid.UUID] = None):
        self.text = text
        self.speaker = speaker
        self.index = index
        self.segment_id = segment_id or uuid.uuid4()

    def __str__(self):
        return f'{self.index}. {self.speaker}: "{self.text}"'

    def to_dict(self) -> dict:
        return {
            "index": int(self.index),
            "speaker": str(self.speaker),
            "text": str(self.text),
            "segment_id": str(self.segment_id)
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptSegment":
        sid = d.get("segment_id")
        return cls(
            text=d.get("text", "") or "",
            speaker=d.get("speaker", "") or "",
            index=int(d.get("index", 0)),
            segment_id=uuid.UUID(sid) if sid else None
        )


class TimeMark:
    def __init__(self, time_ms: int):
        self.time = int(time_ms)

    @staticmethod
    def time_str_to_ms(s: str) -> int:
        hh, mm, rest = s.split(":")
        ss, mmm = rest.split(".")
        return (int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1000 + int(mmm)

    @staticmethod
    def ms_to_time_str(ms: int) -> str:
        ms = int(ms)
        hh = ms // 3600000
        ms %= 3600000
        mm = ms // 60000
        ms %= 60000
        ss = ms // 1000
        mmm = ms % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{mmm:03d}"

    def __str__(self) -> str:
        return self.ms_to_time_str(self.time)

    def to_dict(self) -> dict:
        return {"time_ms": int(self.time)}
    
    @classmethod
    def from_dict(cls, d: Union[dict, int, str]) -> "TimeMark":
        if isinstance(d, dict):
            return cls(int(d.get("time_ms", 0)))
        if isinstance(d, int):
            return cls(d)
        if isinstance(d, str):
            return cls(TimeMark.time_str_to_ms(d))
        return cls(0)


class SourceDocument:
    def __init__(self, file_name: Optional[str] = None, *, meta: Optional[Dict[str, Any]] = None):
        self.file_name = file_name
        self.segments: List[TranscriptSegment] = []
        self.meta: Dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}


class ManualTranscript(SourceDocument):
    def __init__(
        self,
        file_name: Optional[str] = None,
        seperate_sentences: bool = True,
        segments: Optional[List[TranscriptSegment]] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        super().__init__(file_name, meta=meta)

        if segments is not None:
            self.segments = list(segments)
            return

        if not file_name:
            return

        doc = load(file_name)
        paras = doc.getElementsByType(odf_text.P)
        plain = [teletype.extractText(p) for p in paras]

        inx = 0
        for s in plain:
            if not s:
                continue

            if " " not in s:
                speaker, text = "", s
            else:
                speaker, text = s.split(" ", 1)

            if seperate_sentences:
                sentences = [t.strip() for t in text.split(".")]
                for sent in sentences:
                    if sent:
                        self.segments.append(TranscriptSegment(sent, speaker, inx))
                        inx += 1
            else:
                self.segments.append(TranscriptSegment(text.strip(), speaker, inx))
                inx += 1

    def __str__(self):
        return "\n".join(str(s) for s in self.segments)

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "meta": dict(self.meta),
            "segments": [s.to_dict() for s in self.segments]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ManualTranscript":
        if not isinstance(d, dict):
            return cls(file_name=None)

        segs_in = d.get("segments", []) or []
        segs: List[TranscriptSegment] = []
        for i, sd in enumerate(segs_in):
            if isinstance(sd, dict):
                seg = TranscriptSegment.from_dict(sd)
            else:
                seg = TranscriptSegment(text=str(sd), speaker="", index=i)
            segs.append(seg)

        return cls(
            file_name=d.get("file_name"),
            seperate_sentences=False, 
            segments=segs,
            meta=d.get("meta", {}) or {}
        )
    

class ASRExtract(SourceDocument):
    def __init__(
        self,
        file_name: Optional[str] = None,
        segments: Optional[List[TranscriptSegment]] = None,
        times: Optional[List[TimeMark]] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        super().__init__(file_name, meta=meta)
        self.times: List[TimeMark] = []

        if segments is not None:
            self.segments = list(segments)
            self.times = list(times) if times is not None else []
            return

        if not file_name:
            return

        with open(file_name, 'r', encoding='utf-8') as file:
            inx = 0
            for l in file.read().splitlines():
                splitted = l.split('\t')
                if not splitted or splitted[0] == "IN":
                    continue

                if len(splitted) < 3:
                    continue

                self.segments.append(TranscriptSegment(splitted[2], splitted[1], inx))
                self.times.append(TimeMark(TimeMark.time_str_to_ms(splitted[0])))
                inx += 1

    def __str__(self):
        return "\n".join(f"{self.times[i]}\t{self.segments[i]}" for i in range(len(self.segments)))

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "meta": dict(self.meta),
            "segments": [s.to_dict() for s in self.segments],
            "times": [t.to_dict() for t in self.times]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ASRExtract":
        if not isinstance(d, dict):
            return cls(file_name=None)

        segs_in = d.get("segments", []) or []
        segs: List[TranscriptSegment] = []
        for i, sd in enumerate(segs_in):
            if isinstance(sd, dict):
                seg = TranscriptSegment.from_dict(sd)
            else:
                seg = TranscriptSegment(text=str(sd), speaker="", index=i)
            segs.append(seg)

        times_in = d.get("times", []) or []
        times: List[TimeMark] = [TimeMark.from_dict(t) for t in times_in]

        return cls(
            file_name=d.get("file_name"),
            segments=segs,
            times=times,
            meta=d.get("meta", {}) or {}
        )

if __name__ == "__main__":
    extract = ASRExtract("FaPra Timealignment\FaPra Timealignment\ADG3149_01_01_de_speaker.csv")
    print(extract)

    transcript = ManualTranscript("FaPra Timealignment\FaPra Timealignment\ADG3149_01_01.odt")
    print(transcript)
