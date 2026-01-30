from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QSplitter, QGroupBox
)
from PyQt5.QtCore import Qt


class ProjectTab(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)

        self.project = project



        root = QHBoxLayout(self)
        self.splitter_main = QSplitter(Qt.Vertical, self)

        # Top half
        self.top_splitter = QSplitter(Qt.Horizontal, self.splitter_main)

        self.transcript_box = self._build_transcript_box()
        self.asr_box = self._build_asr_box()

        self.top_splitter.addWidget(self.transcript_box)
        self.top_splitter.addWidget(self.asr_box)

        # Bottom half
        self.log_box = self._build_log_box()

        # Combine halfs
        self.splitter_main.addWidget(self.top_splitter)
        self.splitter_main.addWidget(self.log_box)

        root.addWidget(self.splitter_main)
        print("ProjectTab built")

    def _build_transcript_box(self) -> QGroupBox:
        box = QGroupBox("Transcript")
        layout = QVBoxLayout(box)

        self.table_transcript = QTableWidget(0, 3)
        self.table_transcript.setHorizontalHeaderLabels(["Index", "Speaker", "Text"])
        self._configure_table(self.table_transcript)

        self.table_transcript.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        layout.addWidget(self.table_transcript)
        return box
    
    def _build_asr_box(self) -> QGroupBox:
        box = QGroupBox("ASR-Extract")
        layout = QVBoxLayout(box)

        self.table_asr = QTableWidget(0, 2)
        self.table_asr.setHorizontalHeaderLabels(["Time", "Segments"])
        self._configure_table(self.table_asr)

        self.table_asr.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        layout.addWidget(self.table_asr)
        return box
    
    def _build_log_box(self) -> QGroupBox:
        box = QGroupBox("Projekt / Log")
        layout = QVBoxLayout(box)

        self.editor = QTextEdit()
        self.editor.setPlaceholderText("")

        layout.addWidget(self.editor)
        return box

    # ---------- Getters/Setters ----------
    def set_log_text(self, text: str):
        self.editor.setPlainText(text)

    def set_transcript(self, transcript):
        self._fill_transcript_table(transcript)

    def set_asr_extract(self, extract):
        self._fill_asr_table(extract)

    # ---------- Internal helpers ----------
    def _configure_table(self, table: QTableWidget):
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
