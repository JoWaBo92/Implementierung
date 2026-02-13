from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QSplitter, QGroupBox
)
from PyQt5.QtCore import Qt

from domain.project import Project
from utils import BaseTab, TableUtils


class ProjectTab(QWidget, BaseTab):
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._init_base_tab(project, parent)

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

    def _build_transcript_box(self) -> QGroupBox:
        box = QGroupBox("Transcript")
        layout = QVBoxLayout(box)

        self.table_transcript = QTableWidget(0, 3)
        self.table_transcript.setHorizontalHeaderLabels(["Index", "Speaker", "Text"])

        TableUtils.configure_table_basic(self.table_transcript)
        TableUtils.configure_table_fill(
            self.table_transcript,
            resize_cols=[0, 1],   
            text_cols=[2],        
        )

        layout.addWidget(self.table_transcript)
        return box
    
    def _build_asr_box(self) -> QGroupBox:
        box = QGroupBox("ASR-Extract")
        layout = QVBoxLayout(box)

        self.table_asr = QTableWidget(0, 2)
        self.table_asr.setHorizontalHeaderLabels(["Time", "Segments"])
        
        TableUtils.configure_table_basic(self.table_asr)
        TableUtils.configure_table_fill(
            self.table_asr,
            resize_cols=[0],  
            text_cols=[1],   
        )

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

    def set_project(self, project: Project):
        self.project = project
        self.set_transcript(project.transcript)
        self.set_asr_extract(project.asr_extract)

    def set_transcript(self, transcript):
        self._fill_transcript_table(transcript)

    def set_asr_extract(self, extract):
        self._fill_asr_table(extract)

    # ---------- Internal helpers ----------
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


    def _fill_asr_table(self, extract):
        self._clear_table(self.table_asr)

        times = getattr(extract, "times", [])
        segs = getattr(extract, "segments", [])

        n = min(len(times), len(segs))
        self.table_asr.setRowCount(n)

        for r in range(n):
            self.table_asr.setItem(r, 0, QTableWidgetItem(str(times[r])))
            self.table_asr.setItem(r, 1, QTableWidgetItem(str(segs[r])))