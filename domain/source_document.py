import uuid
from odf.opendocument import load
from odf import text as odf_text
from odf import teletype

class SourceDocument:
    def __init__(self, file_name):
        self.file_name = file_name
        self.segments = []

class ManualTranscript(SourceDocument):
    def __init__(self, file_name):
        super().__init__(file_name)

        doc = load(file_name)
        paras = doc.getElementsByType(odf_text.P)
        plain = [teletype.extractText(p) for p in paras]

        inx = 0
        for s in plain:
            if len(s) < 1:
                continue

            speaker, text = s.split(" ", 1)
            self.segments.append(TranscriptSegment(text, speaker, inx))
            inx += 1

    def __str__(self):
        lines = [f"{self.segments[i]}" for i in range(len(self.segments))]
        return '\n'.join(lines)

class ASRExtract(SourceDocument):
    def __init__(self, file_name):
        super().__init__(file_name)

        self.times = []

        with open(file_name, 'r', encoding='utf-8') as file:
            inx = 0
            for l in file.read().splitlines():
                splitted = l.split('\t')

                if splitted[0] == "IN":
                    continue

                self.segments.append(TranscriptSegment(splitted[2], splitted[1], inx))
                inx += 1
                self.times.append(TimeMark(splitted[0]))

    def __str__(self):
        lines = [f"{self.times[i]}\t{self.segments[i]}" for i in range(len(self.segments))]
        return '\n'.join(lines)

class TranscriptSegment:
    def __init__(self, text, speaker, index):
        self.text = text
        self.speaker = speaker
        self.index = index
        self.segment_id = uuid.uuid4()

    def __str__(self):
        return f'{self.index}. {self.speaker}: "{self.text}"'

class TimeMark:
    def __init__(self, time):
        self.time = self.time_str_to_ms(time)

    @staticmethod
    def time_str_to_ms(s: str) -> int:
        hh, mm, rest = s.split(":")
        ss, mmm = rest.split(".")
        return (int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1000 + int(mmm)
    
    def __str__(self) -> str:
        ms = self.time

        hh = ms // 3600000
        ms %= 3600000
        mm = ms // 60000
        ms %= 60000
        ss = ms // 1000
        mmm = ms % 1000

        return f"{hh:02d}:{mm:02d}:{ss:02d}.{mmm:03d}"

if __name__ == "__main__":
    extract = ASRExtract("FaPra Timealignment\FaPra Timealignment\ADG3149_01_01_de_speaker.csv")
    print(extract)

    transcript = ManualTranscript("FaPra Timealignment\FaPra Timealignment\ADG3149_01_01.odt")
    print(transcript)


    

    