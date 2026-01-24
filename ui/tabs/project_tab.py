from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView
)
from PyQt5.QtCore import Qt


class ProjectTab(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)

        self.project = project

        self.editor = QTextEdit()
        self.editor.setPlaceholderText("")

        self.table_transcript = QTableWidget(0, 3)
        self.table_transcript.setHorizontalHeaderLabels(["Index", "Speaker", "Text"])
        self._configure_table(self.table_transcript, fixed_height=220)

        self.table_asr = QTableWidget(0, 2)
        self.table_asr.setHorizontalHeaderLabels(["Time", "Segment"])
        self._configure_table(self.table_asr, fixed_height=220)

        left = QVBoxLayout()
        left.addWidget(QLabel("Transcript"))
        left.addWidget(self.table_transcript)

        right = QVBoxLayout()
        right.addWidget(QLabel("ASR-Extract"))
        right.addWidget(self.table_asr)

        tables_row = QHBoxLayout()
        tables_row.addLayout(left, 1)
        tables_row.addLayout(right, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(tables_row)
        layout.addWidget(QLabel("Projekt / Log"))
        layout.addWidget(self.editor)

    # ---------- Getters/Setters ----------
    def set_log_text(self, text: str):
        self.editor.setPlainText(text)

    def set_transcript(self, transcript):
        self._fill_transcript_table(transcript)

    def set_asr_extract(self, extract):
        self._fill_asr_table(extract)

    # ---------- Internal helpers ----------
    def _configure_table(self, table: QTableWidget, fixed_height: int):
        table.setMinimumHeight(fixed_height)
        table.setMaximumHeight(fixed_height)

        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)

        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.Interactive)

    def _clear_table(self, tbl: QTableWidget):
        tbl.setRowCount(0)

    def _fill_transcript_table(self, transcript):
        self._clear_table(self.table_transcript)

        segments = getattr(transcript, "segments", [])
        self.table_transcript.setRowCount(len(segments))

        for r, seg in enumerate(segments):
            idx = getattr(seg, "index", r)
            speaker = getattr(seg, "speaker", "")
            text = getattr(seg, "text", str(seg))

            self.table_transcript.setItem(r, 0, QTableWidgetItem(str(idx)))
            self.table_transcript.setItem(r, 1, QTableWidgetItem(str(speaker)))
            self.table_transcript.setItem(r, 2, QTableWidgetItem(str(text)))

        self.table_transcript.resizeColumnsToContents()

    def _fill_asr_table(self, extract):
        self._clear_table(self.table_asr)

        times = getattr(extract, "times", [])
        segs = getattr(extract, "segments", [])

        n = min(len(times), len(segs))
        self.table_asr.setRowCount(n)

        for r in range(n):
            self.table_asr.setItem(r, 0, QTableWidgetItem(str(times[r])))
            self.table_asr.setItem(r, 1, QTableWidgetItem(str(segs[r])))

        self.table_asr.resizeColumnsToContents()
