from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QSplitter, QGroupBox, QComboBox, QPushButton, QCheckBox,
    QMessageBox
)
from PyQt5.QtCore import Qt, QTimer

from spacy.util import get_installed_models

from typing import List

from domain.project import Project, PreprocessingResultCollection
from domain.preprocessing import PreprocessingResult, PreprocessingConfig, PreprocessingPipeline

class PreprocessingTab(QWidget):
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)

        self.project = project

        self.results: List[PreprocessingResult] = []

        root = QHBoxLayout(self)

        self.splitter_main = QSplitter(Qt.Vertical, self)
        self.splitter_main.setSizes([1, 3])

        # Top half (controls + history)
        self.top_splitter = QSplitter(Qt.Horizontal, self.splitter_main)

        self.controls_box = self._build_controls_box()
        self.history_box = self._build_history_box()

        self.top_splitter.addWidget(self.controls_box)
        self.top_splitter.addWidget(self.history_box)

        # Bottom half (results)
        self.bottom_splitter = QSplitter(Qt.Horizontal, self.splitter_main)

        self.transcript_result_box, self.table_transcript_result = self._build_result_box("Preprocessing-Ergebnis Transkript")
        self.extract_result_box, self.table_extract_result = self._build_result_box("Preprocessing-Ergebnis ASR-Extrakt")

        self.bottom_splitter.addWidget(self.transcript_result_box)
        self.bottom_splitter.addWidget(self.extract_result_box)

        # Combine sides
        self.splitter_main.addWidget(self.top_splitter)
        self.splitter_main.addWidget(self.bottom_splitter)

        root.addWidget(self.splitter_main)
        self.splitter_main.setStretchFactor(0, 1)
        self.splitter_main.setStretchFactor(1, 2)
        QTimer.singleShot(0, lambda: self.splitter_main.setSizes([200, 400]))

        self._wire_actions()

    def _wire_actions(self):
        self.btn_run.clicked.connect(self._on_click_run)

        self.table_history.selectionModel().selectionChanged.connect(self._on_history_selection_changed)
        self.table_history.cellDoubleClicked.connect(lambda r, c: self._show_history_index(r))

    def _build_controls_box(self) -> QGroupBox:
        box = QGroupBox("Preprocessing konfigurieren")
        layout = QVBoxLayout(box)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Sprachmodell:"))

        self.combo_language_model = QComboBox()
        self._fill_combo_language_model()

        row1.addWidget(self.combo_language_model, 1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(6)

        lbl_options = QLabel("Optionen:")
        lbl_options.setAlignment(Qt.AlignTop)
        row2.addWidget(lbl_options)

        options_col = QVBoxLayout()
        options_col.setContentsMargins(0, 0, 0, 0)
        options_col.setSpacing(2)

        self.chk_lowercase = QCheckBox("In Kleinbuchstaben umwandeln ")
        self.chk_lowercase.setChecked(False)

        self.chk_keep_punct = QCheckBox("Interpunktion behalten")
        self.chk_keep_punct.setChecked(True)

        options_col.addWidget(self.chk_lowercase)
        options_col.addWidget(self.chk_keep_punct)

        row2.addLayout(options_col)
        row2.addStretch(1)
        layout.addLayout(row2)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_run = QPushButton("Preprocessing starten")
        btn_row.addWidget(self.btn_run)
        layout.addLayout(btn_row)

        return box
    
    def _build_history_box(self) -> QGroupBox:
        box = QGroupBox("Vorherige Preprocessings")
        layout = QVBoxLayout(box)

        self.table_history = QTableWidget(0, 4)
        self.table_history.setHorizontalHeaderLabels(["Zeit", "Modell", "Klein", "Punkt"])

        self._configure_table(self.table_history)
        self.table_history.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

        layout.addWidget(self.table_history)
        return box
    
    def _build_result_box(self, title: str) -> tuple[QGroupBox, QTableWidget]:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)

        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(["#", "Original", "Normalisiert", "Tokens"])

        self._configure_table(table)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        layout.addWidget(table)
        return box, table
    
    # ----------- Getters / Setters -----------

    def set_project(self, project: Project):
        self.project = project
        self._fill_all_tables_with_latest()

    # ----------- Internal helpers ------------
    def _configure_table(self, table: QTableWidget):
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)

        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.Stretch)

    def _clear_table(self, table: QTableWidget):
        table.setRowCount(0)

    def _fill_result_table(self, table: QTableWidget, results: List):
        self._clear_table(table)
        table.setRowCount(len(results))

        for r, res in enumerate(results):
            
            raw = getattr(res, "raw_text", "")
            clean = getattr(res, "clean_text", "")
            tokens = getattr(res, "tokens", "")
            token_str = ", ".join(map(str, tokens))

            table.setItem(r, 0, QTableWidgetItem(str(r)))
            table.setItem(r, 1, QTableWidgetItem(str(raw)))
            table.setItem(r, 2, QTableWidgetItem(str(clean)))
            table.setItem(r, 3, QTableWidgetItem(token_str))

    def _fill_history_table(self):
        table = self.table_history
        results: List[PreprocessingResultCollection] = self.project.preprocessing_results

        table.blockSignals(True)
        try:
            self._clear_table(table)
            table.setRowCount(len(results))

            for r, res in enumerate(results):
                time = getattr(res, "time", "")
                model = getattr(res.config, "spacy_model", "")
                lowercase = getattr(res.config, "lowercase", "")
                punct = getattr(res.config, "keep_punct", "")

                table.setItem(r, 0, QTableWidgetItem(str(time)))
                table.setItem(r, 1, QTableWidgetItem(str(model)))
                table.setItem(r, 2, QTableWidgetItem(str(lowercase)))
                table.setItem(r, 3, QTableWidgetItem(str(punct)))
        finally:
            table.blockSignals(False)

        if len(results) > 0:
            last = len(results) - 1
            table.selectRow(last)
            self._show_history_index(last)

    def _fill_all_tables_with_latest(self):
        if len(self.project.preprocessing_results) > 0:
            pre_result_transcript = self.project.preprocessing_results[-1].transcript_results
            pre_result_extract = self.project.preprocessing_results[-1].extract_results
            self._fill_result_table(self.table_transcript_result, pre_result_transcript)
            self._fill_result_table(self.table_extract_result, pre_result_extract)

            self._fill_history_table()

    def _fill_combo_language_model(self):
        models = sorted(get_installed_models())

        self.combo_language_model.blockSignals(True)
        self.combo_language_model.clear()

        if not models:
            self.combo_language_model.addItem("<kein spaCy Modell installiert>")
            self.combo_language_model.setEnabled(False)
            return
        
        self.combo_language_model.setEnabled(True)
        self.combo_language_model.addItems(models)

        preferred = "de_core_news_md"
        if preferred in models:
            self.combo_language_model.setCurrentText(preferred)

        self.combo_language_model.blockSignals(False)

    # ----------- Event handlers --------------

    def on_tab_activated(self):
        self._fill_combo_language_model()

    def _on_click_run(self):
        if not self.combo_language_model.isEnabled():
            QMessageBox.warning(self, "Kein spaCy-Modell installiert.", 
                "Zur Durchführung des Preprocessings bitte mindestens ein spaCy-Sprachmodell installieren")
            return
        
        transcript = getattr(self.project, "transcript", None)
        if not transcript:
            QMessageBox.warning(self, "Kein Transcript vorhanden.",
                "Zur Durchführung des Preprocessings bitte eine Transcript-Datei laden")
            return
        
        extract = getattr(self.project, "asr_extract", None)
        if not extract:
            QMessageBox.warning(self, "Kein ASR-Extrakt vorhanden.",
                "Zur Durchführung des Preprocessings bitte eine ASR-Extrakt-Datei laden")
            return

        pipe_config = PreprocessingConfig(
            spacy_model=self.combo_language_model.currentText(), 
            lowercase=self.chk_lowercase.isChecked(), 
            keep_punct=self.chk_keep_punct.isChecked()
            )
        pipe = PreprocessingPipeline(config=pipe_config)
        pre_result_transcript = pipe.run_batch(self.project.transcript.segments)
        pre_result_extract = pipe.run_batch(self.project.asr_extract.segments)

        result_collection = PreprocessingResultCollection(config=pipe_config)
        result_collection.transcript_results = pre_result_transcript
        result_collection.extract_results = pre_result_extract
        self.project.preprocessing_results.append(result_collection)
        self.project.current.preprocessing = result_collection

        self._fill_result_table(self.table_transcript_result, pre_result_transcript)
        self._fill_result_table(self.table_extract_result, pre_result_extract)

        self._fill_history_table()
        print("Preprocessing done")

    def _on_history_selection_changed(self, selected, deselected):
        rows = self.table_history.selectionModel().selectedRows()
        if not rows:
            return
        self._show_history_index(rows[0].row())
        self.project.current.preprocessing = self.project.preprocessing_results[rows[0].row()]

    def _show_history_index(self, row: int):
        if not self.project or not getattr(self.project, "preprocessing_results", None):
            return

        if row < 0 or row >= len(self.project.preprocessing_results):
            return

        col: PreprocessingResultCollection = self.project.preprocessing_results[row]
        self._fill_result_table(self.table_transcript_result, col.transcript_results)
        self._fill_result_table(self.table_extract_result, col.extract_results)