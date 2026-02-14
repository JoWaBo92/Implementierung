from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QSplitter, QGroupBox, QComboBox, QPushButton, QCheckBox,
    QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor 

import numpy as np

from spacy.util import get_installed_models

from typing import List, Optional

from domain.project import Project, PreprocessingResultCollection, DeviationResultCollection
from domain.preprocessing import PreprocessingResult, PreprocessingConfig, PreprocessingPipeline
from domain.deviation import DeviationCalculator, DeviationAnalysisConfig, DeviationMethod

from utils import BaseTab, TableUtils

class DeviationTab(QWidget, BaseTab):
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._init_base_tab(project, parent)

        self.project = project

        self.results: List[PreprocessingResult] = []
        self.sim_matrix: Optional[np.ndarray] = None   # shape: (n_transcript, n_extract)
        self.current_transcript_index: int = 0
        self._history_results: List[DeviationResultCollection] = []

        root = QHBoxLayout(self)

        self.splitter_main = QSplitter(Qt.Vertical, self)

        # Top half (controls + history)
        self.top_splitter = QSplitter(Qt.Horizontal, self.splitter_main)

        self.controls_box = self._build_controls_box()
        self.history_box = self._build_history_box()

        self.top_splitter.addWidget(self.controls_box)
        self.top_splitter.addWidget(self.history_box)

        # Bottom half (results)
        self.bottom_splitter = QSplitter(Qt.Horizontal, self.splitter_main)

        self.transcript_segment_box = self._build_transcript_segment_box()
        self.extract_result_box = self._build_extract_result_box()

        self.bottom_splitter.addWidget(self.transcript_segment_box)
        self.bottom_splitter.addWidget(self.extract_result_box)

        # Combine sides
        self.splitter_main.addWidget(self.top_splitter)
        self.splitter_main.addWidget(self.bottom_splitter)

        root.addWidget(self.splitter_main)
        self.splitter_main.setStretchFactor(0, 1)
        self.splitter_main.setStretchFactor(1, 2)
        QTimer.singleShot(0, lambda: self.splitter_main.setSizes([200, 400]))

        self._wire_actions()

    # ------------------------------------------------ WIRE --------------------------------------------------------
    def _wire_actions(self):
        self.btn_run.clicked.connect(self._on_click_run)

        self.table_history.selectionModel().selectionChanged.connect(self._on_history_selection_changed)
        self.table_history.cellDoubleClicked.connect(lambda r, c: self._show_history_index(r))

        self.table_transcript.selectionModel().selectionChanged.connect(self._on_transcript_selection_changed)
        self.table_transcript.cellDoubleClicked.connect(lambda r, c: self._show_transcript_index(r))

        self.chk_pos_in_src.toggled.connect(self._toggle_pos_strength)
        self.chk_similar_length.toggled.connect(self.combo_len_strength.setEnabled)

    # ----------------------------------------- BUILD GUI ELEMENTS -------------------------------------------------
    def _build_controls_box(self) -> QGroupBox:
        box = QGroupBox("Abweichungsanalyse konfigurieren")
        layout = QVBoxLayout(box)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Methode:"))

        self.combo_language_model = QComboBox()
        self._fill_combo_method()

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

        self.chk_pos_in_src = QCheckBox("Position in Quellendokument beachten")
        self.chk_pos_in_src.setChecked(False)

        self.chk_similar_length = QCheckBox("Nur ähnlich lange Segmente vergleichen")
        self.chk_similar_length.setChecked(False)

        # --- Position strength dropdown (enabled only if similar_position checked) ---
        row_pos = QHBoxLayout()
        row_pos.addWidget(QLabel("Positionsgewicht:"))

        self.combo_pos_strength = QComboBox()
        self.combo_pos_strength.addItems(["schwach", "mittel", "stark"])
        self.combo_pos_strength.setCurrentText("mittel")
        self.combo_pos_strength.setEnabled(False) 
        row_pos.addWidget(self.combo_pos_strength, 1)

        # --- Length strength dropdown (enabled only if similar_length checked) ---
        row_len_strength = QHBoxLayout()
        row_len_strength.setSpacing(6)
        row_len_strength.addWidget(QLabel("Längenfilter:"))

        self.combo_len_strength = QComboBox()
        self.combo_len_strength.addItems(["schwach", "mittel", "stark"])
        self.combo_len_strength.setCurrentText("schwach")
        self.combo_len_strength.setEnabled(False)
        row_len_strength.addWidget(self.combo_len_strength, 1)

        # --- Add options ---
        options_col.addWidget(self.chk_pos_in_src)
        options_col.addLayout(row_pos)
        options_col.addWidget(self.chk_similar_length)
        options_col.addLayout(row_len_strength)

        row2.addLayout(options_col)
        row2.addStretch(1)
        layout.addLayout(row2)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_run = QPushButton("Abweichungen berechnen")
        btn_row.addWidget(self.btn_run)
        layout.addLayout(btn_row)

        return box
    
    def _build_history_box(self) -> QGroupBox:
        box = QGroupBox("Vorherige Abweichungsanalysen")
        layout = QVBoxLayout(box)

        self.table_history = QTableWidget(0, 4)
        self.table_history.setHorizontalHeaderLabels(["Zeit", "Methode", "Länge", "Position"])

        TableUtils.configure_table_basic(self.table_history)
        TableUtils.configure_table_fill(
            self.table_history,
            resize_cols=[2, 3],  
            text_cols=[0, 1],    
        )

        layout.addWidget(self.table_history)
        return box
    
    def _build_transcript_segment_box(self) -> QGroupBox:
        box = QGroupBox("Transkript-Segment")
        layout = QVBoxLayout(box)

        self.table_transcript = QTableWidget(0, 2)
        self.table_transcript.setHorizontalHeaderLabels(["#", "Normalisiert"])

        TableUtils.configure_table_basic(self.table_transcript)
        TableUtils.configure_table_fill(
            self.table_transcript,
            resize_cols=[0],  
            text_cols=[1],    
        )

        layout.addWidget(self.table_transcript)
        return box
    
    def _build_extract_result_box(self) -> QGroupBox:
        box = QGroupBox("Ähnlichkeit zu ASR-Extrakt-Elementen")
        layout = QVBoxLayout(box)

        self.table_extract = QTableWidget(0, 3)
        self.table_extract.setHorizontalHeaderLabels(["#", "Normalisiert", "Ähnlichkeit"])

        TableUtils.configure_table_basic(self.table_extract)
        TableUtils.configure_table_fill(
            self.table_extract,
            resize_cols=[0, 2],  
            text_cols=[1],          
        )

        layout.addWidget(self.table_extract)
        return box
    
    # ----------- Getters / Setters -----------

    def set_project(self, project: Project):
        self.project = project
        self._fill_all_tables_with_latest()

    # ----------- Internal helpers ------------

    def _toggle_pos_strength(self):
        self.combo_pos_strength.setEnabled(self.chk_pos_in_src.isChecked())

    def _configure_table(self, table: QTableWidget):
        TableUtils.configure_table_basic(table)
        TableUtils.configure_all_interactive(table)

    def _clear_table(self, table: QTableWidget):
        table.setRowCount(0)

    def _get_filtered_deviation_results(self) -> List[DeviationResultCollection]:
        results: List[DeviationResultCollection] = []
        if self.project and getattr(self.project, "deviation_analysis_results", None):
            results = list(self.project.deviation_analysis_results)

        cur_pre = getattr(getattr(self.project, "current", None), "preprocessing", None)
        if cur_pre is None:
            return results

        pre_id = getattr(cur_pre, "result_id", None)
        if pre_id is None:
            return results

        return [r for r in results if getattr(r, "preprocessing_id", None) == pre_id]

    def _fill_transcript_segment_table(self, table: QTableWidget, segments: List):
        self._clear_table(table)
        table.setRowCount(len(segments))

        for r, res in enumerate(segments):
            clean = getattr(res, "clean_text", "")

            table.setItem(r, 0, QTableWidgetItem(str(r)))
            table.setItem(r, 1, QTableWidgetItem(str(clean)))

    def _fill_extract_segment_table(self, table: QTableWidget, segments: List, similarity_col: np.ndarray):
        self._clear_table(table)
        table.setRowCount(len(segments))

        if similarity_col is not None and len(similarity_col) > 0:
            sims = np.asarray(similarity_col, dtype=float)
            finite = np.isfinite(sims)
            vmin = float(np.min(sims[finite])) if np.any(finite) else 0.0
            vmax = float(np.max(sims[finite])) if np.any(finite) else 1.0
        else:
            sims = None
            vmin, vmax = 0.0, 1.0

        def _lerp(a: int, b: int, t: float) -> int:
            return int(round(a + (b - a) * t))

        def _color_for_sim(x: float) -> QColor:
            if not np.isfinite(x) or vmax == vmin:
                return QColor(220, 220, 220) 

            t = (x - vmin) / (vmax - vmin)
            t = max(0.0, min(1.0, float(t)))

            if t < 1/3:
                u = t / (1/3)
                r = _lerp(220, 255, u)
                g = _lerp(0,   165, u)
                b = 0
            elif t < 2/3:
                u = (t - 1/3) / (1/3)
                r = 255
                g = _lerp(165, 255, u)
                b = 0
            else:
                u = (t - 2/3) / (1/3)
                r = _lerp(255, 0,   u)
                g = _lerp(255, 200, u)
                b = 0

            return QColor(r, g, b)

        for r, res in enumerate(segments):
            clean = getattr(res, "clean_text", "")
            sim_val = float(similarity_col[r]) if similarity_col is not None else float("nan")

            table.setItem(r, 0, QTableWidgetItem(str(r)))
            table.setItem(r, 1, QTableWidgetItem(str(clean)))

            item_sim = QTableWidgetItem(f"{sim_val:.4f}")
            item_sim.setBackground(_color_for_sim(sim_val)) 
            table.setItem(r, 2, item_sim)

    def _fill_history_table(self):
        table = self.table_history
        results: List[DeviationResultCollection] = self._get_filtered_deviation_results()
        self._history_results = results

        def _position_label(config: DeviationAnalysisConfig) -> str:
            if not getattr(config, "similar_position", False):
                return "Aus"

            gamma = getattr(config, "position_gamma", None)
            if gamma is None:
                return "An"

            if gamma < 3.0:
                return "Schwach"
            elif gamma < 6.0:
                return "Mittel"
            else:
                return "Stark"
            
        def _length_label(cfg: DeviationAnalysisConfig) -> str:
            if not getattr(cfg, "similar_length", False):
                return "Aus"
            mr = float(getattr(cfg, "length_min_ratio", 0.0))
            a = float(getattr(cfg, "length_alpha", 0.0))

            # Preset reverse-mapping (tolerant)
            def close(x: float, y: float, tol: float = 1e-6) -> bool:
                return abs(x - y) <= tol

            if close(mr, 0.20) and close(a, 0.75):
                return "Schwach"
            if close(mr, 0.35) and close(a, 1.50):
                return "Mittel"
            if close(mr, 0.50) and close(a, 2.50):
                return "Stark"
            return "Benutzerdef."

        table.blockSignals(True)
        try:
            self._clear_table(table)
            table.setRowCount(len(results))

            for r, res in enumerate(results):
                time = getattr(res, "time", "")
                method = getattr(res.config.method, "value", "")
                length = _length_label(res.config)
                pos = _position_label(res.config)

                table.setItem(r, 0, QTableWidgetItem(str(time)))
                table.setItem(r, 1, QTableWidgetItem(str(method)))
                table.setItem(r, 2, QTableWidgetItem(str(length)))
                table.setItem(r, 3, QTableWidgetItem(str(pos)))
        finally:
            table.blockSignals(False)

        if len(results) > 0:
            idx = len(results) - 1
            cur = getattr(getattr(self.project, "current", None), "deviation_analysis", None)
            if cur in results:
                idx = results.index(cur)
            table.selectRow(idx)
            self._show_history_index(idx)

    def _fill_all_tables_with_latest(self):
        if not self.project:
            self._clear_table(self.table_transcript)
            self._clear_table(self.table_extract)
            self._clear_table(self.table_history)
            return

        results = self._get_filtered_deviation_results()
        if len(results) == 0:
            if getattr(self.project, "current", None) is not None:
                self.project.current.deviation_analysis = None
                self.project.current.synchronization = None
            self.sim_matrix = None
            self._clear_table(self.table_transcript)
            self._clear_table(self.table_extract)
            self._fill_history_table()
            return

        cur = getattr(getattr(self.project, "current", None), "deviation_analysis", None)
        selected: DeviationResultCollection = cur if (cur in results) else results[-1]

        if getattr(self.project, "current", None) is not None and getattr(self.project.current, "deviation_analysis", None) is None:
             setattr(self.project.current, "deviation_analysis", selected)

        if getattr(self.project, "current", None) is not None and getattr(self.project.current, "deviation_analysis", None) not in results:
            setattr(self.project.current, "deviation_analysis", selected)

        self.sim_matrix = selected.result_matrix
        self.current_transcript_index = 0

        self._fill_transcript_segment_table(self.table_transcript, selected.transcript_preprocessed)

        if self.sim_matrix is not None and self.sim_matrix.size > 0 and len(selected.extract_preprocessed) > 0:
            self._fill_extract_segment_table(self.table_extract, selected.extract_preprocessed, self.sim_matrix[self.current_transcript_index, :])
            self.table_transcript.selectRow(0)
        else:
            self._clear_table(self.table_extract)

        self._fill_history_table()

    def _fill_combo_method(self):
        methods = []
        if self.project.current.preprocessing is not None:
            if "trf" in self.project.current.preprocessing.config.spacy_model:
                methods.append(DeviationMethod.TRF.value)
            else:
                methods.append(DeviationMethod.STANDARD.value)
            methods.append(DeviationMethod.SENT_TRF.value)

        self.combo_language_model.blockSignals(True)
        self.combo_language_model.clear()

        if not methods:
            self.combo_language_model.addItem("<bisher kein Preprocessing durchgeführt>")
            self.combo_language_model.setEnabled(False)
            return
        
        self.combo_language_model.setEnabled(True)
        self.combo_language_model.addItems(methods)

        preferred = "de_core_news_md"
        if preferred in methods:
            self.combo_language_model.setCurrentText(preferred)

        self.combo_language_model.blockSignals(False)

    # ------------------------------------------------------------- Event handlers ------------------------------------------------------------

    def on_tab_activated(self):
        self._fill_combo_method()

    def _on_click_run(self):
        if not self.combo_language_model.isEnabled():
            QMessageBox.warning(
                self,
                "Keine Preprocessing-Ergebnisse vorhanden.",
                "Zur Durchführung der Abweichungsanalyse zunächst ein Preprocessing durchführen.",
            )
            return

        # Build config from GUI (GUI thread)
        if self.combo_language_model.currentText() == DeviationMethod.SENT_TRF.value:
            library = "paraphrase-multilingual-MiniLM-L12-v2"
            method = DeviationMethod.SENT_TRF
        else:
            library = self.project.current.preprocessing.config.spacy_model
            method = (
                DeviationMethod.TRF
                if self.combo_language_model.currentText() == DeviationMethod.TRF.value
                else DeviationMethod.STANDARD
            )

        gamma_map = {"schwach": 2.5, "mittel": 4.0, "stark": 7.0}
        gamma = float(gamma_map.get(self.combo_pos_strength.currentText(), 4.0))

        len_preset = self.combo_len_strength.currentText().lower()
        len_map = {
            "schwach": dict(min_ratio=0.20, alpha=0.75, ultra_short_tokens=2),
            "mittel": dict(min_ratio=0.35, alpha=1.50, ultra_short_tokens=2),
            "stark": dict(min_ratio=0.50, alpha=2.50, ultra_short_tokens=2),
        }
        lp = len_map.get(len_preset, len_map["mittel"])

        config = DeviationAnalysisConfig(
            library=library,
            method=method,
            similar_length=self.chk_similar_length.isChecked(),
            length_min_ratio=lp["min_ratio"],
            length_alpha=lp["alpha"],
            ultra_short_tokens=lp["ultra_short_tokens"],
            similar_position=self.chk_pos_in_src.isChecked(),
            position_gamma=gamma,
        )

        transcript_results = list(self.project.current.preprocessing.transcript_results)
        extract_results = list(self.project.current.preprocessing.extract_results)
        transcript_docs = [r.doc for r in transcript_results]
        extract_docs = [r.doc for r in extract_results]

        def work():
            calc = DeviationCalculator(config)
            sim_matrix = calc.similarity_matrix(transcript_docs, extract_docs)
            return sim_matrix

        def on_success(sim_matrix):
            self.sim_matrix = sim_matrix

            result_collection = DeviationResultCollection(config=config)
            result_collection.result_matrix = self.sim_matrix
            result_collection.extract_preprocessed = extract_results
            result_collection.transcript_preprocessed = transcript_results
            result_collection.preprocessing_id = self.project.current.preprocessing.result_id

            self.project.deviation_analysis_results.append(result_collection)
            self.project.current.deviation_analysis = result_collection

            self._fill_transcript_segment_table(self.table_transcript, transcript_results)

            self.current_transcript_index = 0
            if len(transcript_results) > 0:
                self.table_transcript.selectRow(0)
            self.current_transcript_index = 0
            if len(transcript_results) > 0:
                self.table_transcript.selectRow(0)

            if self.sim_matrix is not None and self.sim_matrix.size > 0 and len(extract_results) > 0:
                sim_row = self.sim_matrix[self.current_transcript_index, :]
                self._fill_extract_segment_table(self.table_extract, extract_results, sim_row)
            else:
                self._clear_table(self.table_extract)

            self._fill_history_table()
            print("Deviation analysis done")
            self._fill_history_table()
            print("Deviation analysis done")

        self.run_in_worker(
            work,
            on_success=on_success,
            busy_widget=self.btn_run,
            status_msg="Abweichungsanalyse läuft…",
        )

    def _on_history_selection_changed(self, selected, deselected):
        rows = self.table_history.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        self._show_history_index(row)
        if 0 <= row < len(self._history_results):
            self.project.current.deviation_analysis = self._history_results[row]
            self.project.current.synchronization = None
            win = self.window()
            if hasattr(win, "tab_alignment"):
                win.tab_alignment._fill_all_tables_with_latest()

    def _show_history_index(self, row: int):
        if not self.project:
            return

        if row < 0 or row >= len(self._history_results):
            return

        col: DeviationResultCollection = self._history_results[row]
        self.sim_matrix = col.result_matrix

        if getattr(self.project, "current", None) is not None:
            self.project.current.deviation_analysis = col

        self._fill_transcript_segment_table(self.table_transcript, col.transcript_preprocessed)
        self._fill_extract_segment_table(self.table_extract, col.extract_preprocessed, col.result_matrix[self.current_transcript_index, :])

    def _on_transcript_selection_changed(self, selected, deselected):
        rows = self.table_transcript.selectionModel().selectedRows()
        if not rows:
            return
        self._show_transcript_index(rows[0].row())

    def _show_transcript_index(self, row: int):
        if self.sim_matrix is None:
            return

        transcript_results = self.project.current.preprocessing.transcript_results
        extract_results = self.project.current.preprocessing.extract_results

        if row < 0 or row >= len(transcript_results):
            return

        self.current_transcript_index = row

        if len(extract_results) == 0:
            self._clear_table(self.table_extract)
            return

        sim_row = self.sim_matrix[row, :]  # 1D
        self._fill_extract_segment_table(self.table_extract, extract_results, sim_row)