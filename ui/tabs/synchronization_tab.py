from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QSplitter, QGroupBox, QComboBox, QPushButton, QCheckBox, QSpinBox,
    QMessageBox
)
from PyQt5.QtCore import Qt, QTimer

import numpy as np
from typing import List, Optional, Any

from domain.project import Project, SynchronizationResultCollection
from domain.synchronization import SynchronizationCalculator, SynchronizationConfig

from utils import BaseTab, TableUtils


class SynchronizationTab(QWidget, BaseTab):
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._init_base_tab(project, parent)

        self.project = project

        self.results: List[SynchronizationResultCollection] = []                 
        self.sim_matrix: Optional[np.ndarray] = None 
        self.align_path: Optional[list] = None       
        self.align_ranges_by_transcript: Optional[list] = None  
        self.current_transcript_index: int = 0
        self._history_results: List[SynchronizationResultCollection] = []

        root = QHBoxLayout(self)

        self.splitter_main = QSplitter(Qt.Vertical, self)

        # Top half (controls + history)
        self.top_splitter = QSplitter(Qt.Horizontal, self.splitter_main)
        self.controls_box = self._build_controls_box()
        self.history_box = self._build_history_box()
        self.top_splitter.addWidget(self.controls_box)
        self.top_splitter.addWidget(self.history_box)

        # Bottom half (transcript + aligned ASR)
        self.bottom_splitter = QSplitter(Qt.Horizontal, self.splitter_main)
        self.transcript_box = self._build_transcript_box()
        self.aligned_extract_box = self._build_aligned_extract_box()
        self.bottom_splitter.addWidget(self.transcript_box)
        self.bottom_splitter.addWidget(self.aligned_extract_box)

        # Combine
        self.splitter_main.addWidget(self.top_splitter)
        self.splitter_main.addWidget(self.bottom_splitter)
        root.addWidget(self.splitter_main)

        self.splitter_main.setStretchFactor(0, 1)
        self.splitter_main.setStretchFactor(1, 2)
        QTimer.singleShot(0, lambda: self.splitter_main.setSizes([200, 400]))

        self._wire_actions()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def set_project(self, project: Project):
        self.project = project
        self._fill_all_tables_with_latest()

    def on_tab_activated(self):
        # Tab wurde aktiviert: UI an aktuellen Projektzustand anpassen
        self._fill_all_tables_with_latest()

    # ---------------------------------------------------------------------
    # Wiring
    # ---------------------------------------------------------------------

    def _wire_actions(self):
        self.btn_run.clicked.connect(self._on_click_run)

        self.table_history.selectionModel().selectionChanged.connect(self._on_history_selection_changed)
        self.table_history.cellDoubleClicked.connect(lambda r, c: self._show_history_index(r))

        self.table_transcript.selectionModel().selectionChanged.connect(self._on_transcript_selection_changed)
        self.table_transcript.cellDoubleClicked.connect(lambda r, c: self._show_transcript_index(r))

    # ---------------------------------------------------------------------
    # UI Builders
    # ---------------------------------------------------------------------

    def _build_controls_box(self) -> QGroupBox:
        box = QGroupBox("Synchronisierung konfigurieren")
        layout = QVBoxLayout(box)

        # Row: alignment algorithm
        row_alg = QHBoxLayout()
        row_alg.addWidget(QLabel("Verfahren:"))

        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItems(["DTW (Sequenz-Alignment)"])
        row_alg.addWidget(self.combo_algorithm, 1)
        layout.addLayout(row_alg)

        # Options column
        row_opts = QHBoxLayout()
        row_opts.setSpacing(6)

        lbl_options = QLabel("Optionen:")
        lbl_options.setAlignment(Qt.AlignTop)
        row_opts.addWidget(lbl_options)

        options_col = QVBoxLayout()
        options_col.setContentsMargins(0, 0, 0, 0)
        options_col.setSpacing(6)

        # ASR split tolerance (vertical step penalty)
        row_v = QHBoxLayout()
        row_v.addWidget(QLabel("ASR-Split-Toleranz:"))
        self.combo_asr_split = QComboBox()
        self.combo_asr_split.addItems(["schwach", "mittel", "stark"])
        self.combo_asr_split.setCurrentText("mittel")
        row_v.addWidget(self.combo_asr_split, 1)
        options_col.addLayout(row_v)

        # Transcript merge tolerance (horizontal step penalty)
        row_h = QHBoxLayout()
        row_h.addWidget(QLabel("Transkript-Zusammenfassung:"))
        self.combo_tr_merge = QComboBox()
        self.combo_tr_merge.addItems(["schwach", "mittel", "stark"])
        self.combo_tr_merge.setCurrentText("mittel")
        row_h.addWidget(self.combo_tr_merge, 1)
        options_col.addLayout(row_h)

        # Minimum similarity gate
        row_min = QHBoxLayout()
        row_min.addWidget(QLabel("Mindestähnlichkeit:"))
        self.combo_min_sim = QComboBox()
        self.combo_min_sim.addItems(["niedrig", "mittel", "hoch"])
        self.combo_min_sim.setCurrentText("mittel")
        row_min.addWidget(self.combo_min_sim, 1)
        options_col.addLayout(row_min)

        # Band limiting
        row_band = QHBoxLayout()
        self.chk_use_band = QCheckBox("Bandbegrenzung (Sakoe–Chiba)")
        self.chk_use_band.setChecked(False)
        row_band.addWidget(self.chk_use_band)

        row_band.addWidget(QLabel("± Segmente:"))
        self.spin_band = QSpinBox()
        self.spin_band.setRange(0, 10000)
        self.spin_band.setValue(50)
        self.spin_band.setEnabled(False)
        row_band.addWidget(self.spin_band)

        row_band.addStretch(1)
        options_col.addLayout(row_band)

        self.chk_use_band.toggled.connect(self.spin_band.setEnabled)

        row_opts.addLayout(options_col, 1)
        row_opts.addStretch(1)
        layout.addLayout(row_opts)

        # Start button
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_run = QPushButton("Synchronisieren")
        btn_row.addWidget(self.btn_run)
        layout.addLayout(btn_row)

        return box

    def _build_history_box(self) -> QGroupBox:
        box = QGroupBox("Vorherige Synchronisierungen")
        layout = QVBoxLayout(box)

        self.table_history = QTableWidget(0, 4)
        self.table_history.setHorizontalHeaderLabels(["Zeit", "Verfahren", "Optionen", "Qualität"])

        TableUtils.configure_table_basic(self.table_history)
        TableUtils.configure_table_fill(
            self.table_history,
            resize_cols=[],
            text_cols=[0, 1, 2, 3],
        )

        layout.addWidget(self.table_history)
        return box

    def _build_transcript_box(self) -> QGroupBox:
        box = QGroupBox("Transkript (Alignment-Übersicht)")
        layout = QVBoxLayout(box)

        self.table_transcript = QTableWidget(0, 6)
        self.table_transcript.setHorizontalHeaderLabels(["#", "Normalisiert", "ASR-Bereich", "Ø-Score", "Start", "Ende"])
        
        TableUtils.configure_table_basic(self.table_transcript)
        TableUtils.configure_table_fill(
            self.table_transcript,
            resize_cols=[0, 2, 3, 4, 5],
            text_cols=[1],
        )

        layout.addWidget(self.table_transcript)
        return box

    def _build_aligned_extract_box(self) -> QGroupBox:
        box = QGroupBox("Zugeordnete ASR-Segmente")
        layout = QVBoxLayout(box)

        self.table_extract = QTableWidget(0, 5)
        self.table_extract.setHorizontalHeaderLabels(["#", "Normalisiert", "Similarity", "Pfad", "Zeit"])

        TableUtils.configure_table_basic(self.table_extract)
        TableUtils.configure_table_fill(
            self.table_extract,
            resize_cols=[0, 2, 3, 4],
            text_cols=[1],
        )

        layout.addWidget(self.table_extract)
        return box


    # ---------------------------------------------------------------------
    # Table helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _ms_to_time_str(ms: int) -> str:
        hh = ms // 3600000
        ms %= 3600000
        mm = ms // 60000
        ms %= 60000
        ss = ms // 1000
        mmm = ms % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{mmm:03d}"


    def _clear_table(self, table: QTableWidget):
        table.setRowCount(0)

    def _get_filtered_sync_results(self) -> List[SynchronizationResultCollection]:

        all_results = list(getattr(self.project, "synchronization_results", []) or [])
        cur_dev = getattr(getattr(self.project, "current", None), "deviation_analysis", None)
        if cur_dev is None:
            return []
        dev_id = getattr(cur_dev, "result_id", None)
        if dev_id is None:
            return []
        return [r for r in all_results if getattr(r, "deviation_id", None) == dev_id]

    # ---------------------------------------------------------------------
    # Fillers (implemented step-by-step later)
    # ---------------------------------------------------------------------

    def _fill_history_table(self):
        table = self.table_history

        results = self._get_filtered_sync_results()
        self._history_results = results

        table.blockSignals(True)
        try:
            self._clear_table(table)
            table.setRowCount(len(results))

            for r, res in enumerate(results):
                # --- time ---
                t = getattr(res, "time", "")
                if hasattr(t, "strftime"):
                    t_str = t.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    t_str = str(t)

                # --- algorithm ---
                cfg = getattr(res, "config", None)
                alg = getattr(cfg, "algorithm", None)
                alg_str = getattr(alg, "value", None) if alg is not None else None
                if not alg_str:
                    alg_str = "DTW"

                # --- options ---
                step_v = getattr(cfg, "step_v", None)
                step_h = getattr(cfg, "step_h", None)
                min_sim = getattr(cfg, "min_sim", None)
                band = getattr(cfg, "band", None)

                opt_parts = []
                if step_v is not None:
                    opt_parts.append(f"v={float(step_v):.2f}")
                if step_h is not None:
                    opt_parts.append(f"h={float(step_h):.2f}")
                if min_sim is not None:
                    opt_parts.append(f"min={float(min_sim):.2f}")
                opt_parts.append(f"band={band}" if band is not None else "band=aus")
                opt_str = ", ".join(opt_parts)

                # --- quality ---
                mean_sim = getattr(res, "mean_similarity_on_path", None)
                total_cost = getattr(res, "total_cost", None)

                q_parts = []
                if mean_sim is not None:
                    q_parts.append(f"Øsim={float(mean_sim):.3f}")
                if total_cost is not None:
                    q_parts.append(f"cost={float(total_cost):.1f}")
                q_str = " | ".join(q_parts) if q_parts else ""

                table.setItem(r, 0, QTableWidgetItem(t_str))
                table.setItem(r, 1, QTableWidgetItem(str(alg_str)))
                table.setItem(r, 2, QTableWidgetItem(opt_str))
                table.setItem(r, 3, QTableWidgetItem(q_str))
        finally:
            table.blockSignals(False)

        # Auto-select current (falls gesetzt), sonst letztes Ergebnis
        # Auto-select current (falls gesetzt), sonst letztes Ergebnis
        if len(results) > 0:
            idx = len(results) - 1
            cur = getattr(getattr(self.project, "current", None), "synchronization", None)
            if cur in results:
                idx = results.index(cur)
            table.selectRow(idx)
            self._show_history_index(idx)
            idx = len(results) - 1
            cur = getattr(getattr(self.project, "current", None), "synchronization", None)
            if cur in results:
                idx = results.index(cur)
            table.selectRow(idx)
            self._show_history_index(idx)


    def _fill_transcript_table(self, segments: List[Any]):
        table = self.table_transcript

        col = None
        if self.project and getattr(self.project, "current", None) is not None:
            col = getattr(self.project.current, "synchronization", None)
        if col is None and self.project and getattr(self.project, "synchronization_results", None):
            if len(self.project.synchronization_results) > 0:
                col = self.project.synchronization_results[-1]

        aligned = getattr(col, "aligned_transcript", None) if col is not None else None

        table.blockSignals(True)
        try:
            self._clear_table(table)
            table.setRowCount(len(segments))

            for i, seg in enumerate(segments):
                clean = getattr(seg, "clean_text", "")
                table.setItem(i, 0, QTableWidgetItem(str(i)))
                table.setItem(i, 1, QTableWidgetItem(str(clean)))

                range_str = "—"
                mean_str = "—"
                start_str = "—"
                end_str = "—"

                if aligned is not None and 0 <= i < len(aligned):
                    a = aligned[i]
                    if a.extract_j0 >= 0 and a.extract_j1 >= 0:
                        range_str = f"{a.extract_j0}–{a.extract_j1}"
                    if a.mean_similarity is not None:
                        mean_str = f"{a.mean_similarity:.3f}"
                    if a.start_ms is not None:
                        start_str = self._ms_to_time_str(int(a.start_ms))
                    if a.end_ms is not None:
                        end_str = self._ms_to_time_str(int(a.end_ms))

                table.setItem(i, 2, QTableWidgetItem(range_str))
                table.setItem(i, 3, QTableWidgetItem(mean_str))
                table.setItem(i, 4, QTableWidgetItem(start_str))
                table.setItem(i, 5, QTableWidgetItem(end_str))
        finally:
            table.blockSignals(False)

        if len(segments) > 0:
            self.current_transcript_index = max(0, min(self.current_transcript_index, len(segments) - 1))
            table.selectRow(self.current_transcript_index)


    def _fill_aligned_extract_table(self, segments: List[Any], sim_row: Optional[np.ndarray] = None):
        table = self.table_extract
        i = int(getattr(self, "current_transcript_index", 0))

        # --- resolve ranges/path/sim ---
        ranges = self.align_ranges_by_transcript
        path = self.align_path
        sim = self.sim_matrix

        current_sync = None
        if self.project and getattr(self.project, "current", None) is not None:
            current_sync = getattr(self.project.current, "synchronization", None)

        if ranges is None and current_sync is not None:
            ranges = getattr(current_sync, "alignment_ranges_by_transcript", None)
        if path is None and current_sync is not None:
            path = getattr(current_sync, "alignment_path", None)
        if sim is None and current_sync is not None:
            sim = getattr(current_sync, "similarity_matrix", None)

        if (ranges is None or path is None) and self.project and getattr(self.project, "synchronization_results", None):
            if len(self.project.synchronization_results) > 0:
                last = self.project.synchronization_results[-1]
                if ranges is None:
                    ranges = getattr(last, "alignment_ranges_by_transcript", None)
                if path is None:
                    path = getattr(last, "alignment_path", None)
                if sim is None:
                    sim = getattr(last, "similarity_matrix", None)

        # --- compute range for current transcript index ---
        j0, j1 = (-1, -1)
        if ranges is not None and 0 <= i < len(ranges):
            j0, j1 = ranges[i]

        # If no valid range -> clear table and return
        if j0 is None or j1 is None or j0 < 0 or j1 < 0 or j0 > j1 or len(segments) == 0:
            self._clear_table(table)
            return

        j0 = max(0, int(j0))
        j1 = min(len(segments) - 1, int(j1))

        # --- extract times from project ---
        extract_times_ms = None
        if self.project is not None and getattr(self.project, "asr_extract", None) is not None:
            # ensure same length as extract segments
            extract_times_ms = [tm.time for tm in self.project.asr_extract.times]

        # --- build move lookup from path: dict[(i,j)] -> move ---
        moves = {}
        if path:
            for (i0, j0p), (i1, j1p) in zip(path[:-1], path[1:]):
                if i1 == i0 + 1 and j1p == j0p + 1:
                    moves[(i1, j1p)] = "diag"
                elif i1 == i0 + 1 and j1p == j0p:
                    moves[(i1, j1p)] = "up"      # transcript advanced (n:1)
                elif i1 == i0 and j1p == j0p + 1:
                    moves[(i1, j1p)] = "left"    # extract advanced (1:n)

            moves[path[0]] = "start"

        def _move_label(m: str) -> str:
            if m == "diag":
                return "↘"
            if m == "left":
                return "→"
            if m == "up":
                return "↓"
            if m == "start":
                return "S"
            return ""

        # --- fill table ---
        table.blockSignals(True)
        try:
            self._clear_table(table)
            n_rows = (j1 - j0 + 1)
            table.setRowCount(n_rows)

            for r, j in enumerate(range(j0, j1 + 1)):
                clean = getattr(segments[j], "clean_text", "")

                # similarity
                sim_val = None
                if sim_row is not None:
                    if 0 <= j < len(sim_row):
                        sim_val = float(sim_row[j])
                elif isinstance(sim, np.ndarray) and sim.ndim == 2:
                    if i < sim.shape[0] and j < sim.shape[1]:
                        sim_val = float(sim[i, j])

                sim_str = f"{sim_val:.3f}" if (sim_val is not None and np.isfinite(sim_val)) else "—"

                # path move symbol
                m = moves.get((i, j), "")
                m_str = _move_label(m)

                # time string (ASR start time)
                if extract_times_ms is not None and 0 <= j < len(extract_times_ms):
                    time_str = self._ms_to_time_str(int(extract_times_ms[j]))
                else:
                    time_str = "—"

                table.setItem(r, 0, QTableWidgetItem(str(j)))
                table.setItem(r, 1, QTableWidgetItem(str(clean)))
                table.setItem(r, 2, QTableWidgetItem(sim_str))
                table.setItem(r, 3, QTableWidgetItem(m_str))
                table.setItem(r, 4, QTableWidgetItem(time_str))
        finally:
            table.blockSignals(False)



    def _fill_all_tables_with_latest(self):
        # --- guards ---
        if not self.project:
            self._clear_table(self.table_history)
            self._clear_table(self.table_transcript)
            self._clear_table(self.table_extract)
            return

        results = self._get_filtered_sync_results()
        self._history_results = results
        
        if len(results) == 0:
            if getattr(self.project, "current", None) is not None:
                self.project.current.synchronization = None
            self._clear_table(self.table_history)
            self._clear_table(self.table_transcript)
            self._clear_table(self.table_extract)
            return

        cur = getattr(getattr(self.project, "current", None), "synchronization", None)
        selected = cur if (cur in results) else results[-1]
        # Ergebnis wählen: project.current.synchronization bevorzugen, sonst letztes Ergebnis
        cur = getattr(getattr(self.project, "current", None), "synchronization", None)
        selected = cur if (cur in results) else results[-1]

        if getattr(self.project, "current", None) is not None and getattr(self.project.current, "synchronization", None) is None:
            setattr(self.project.current, "synchronization", selected)
        # update project.current (nur setzen, wenn noch nichts gewählt)
        if getattr(self.project, "current", None) is not None and getattr(self.project.current, "synchronization", None) is None:
            setattr(self.project.current, "synchronization", selected)

        if getattr(self.project, "current", None) is not None and getattr(self.project.current, "synchronization", None) not in results:
            setattr(self.project.current, "synchronization", selected)

        # cache locally
        self.results = results
        self.sim_matrix = getattr(selected, "similarity_matrix", None)
        self.align_path = getattr(selected, "alignment_path", None)
        self.align_ranges_by_transcript = getattr(selected, "alignment_ranges_by_transcript", None)
        self.sim_matrix = getattr(selected, "similarity_matrix", None)
        self.align_path = getattr(selected, "alignment_path", None)
        self.align_ranges_by_transcript = getattr(selected, "alignment_ranges_by_transcript", None)

        transcript_results = getattr(selected, "transcript_preprocessed", None) or []
        extract_results = getattr(selected, "extract_preprocessed", None) or []
        transcript_results = getattr(selected, "transcript_preprocessed", None) or []
        extract_results = getattr(selected, "extract_preprocessed", None) or []

        # fill tables
        self._fill_history_table()
        self._fill_transcript_table(transcript_results)

        if len(transcript_results) == 0:
            self._clear_table(self.table_extract)
            return

        # keep current index if possible, else default to 0
        self.current_transcript_index = max(0, min(self.current_transcript_index, len(transcript_results) - 1))

        # select transcript row (will NOT trigger show handler reliably if signals blocked elsewhere)
        self.table_transcript.selectRow(self.current_transcript_index)
        self._show_transcript_index(self.current_transcript_index)
        self._show_transcript_index(self.current_transcript_index)

        # fill aligned extract table
        sim_row = None
        if isinstance(self.sim_matrix, np.ndarray) and self.sim_matrix.ndim == 2:
            if self.current_transcript_index < self.sim_matrix.shape[0]:
                sim_row = self.sim_matrix[self.current_transcript_index, :]

        self._fill_aligned_extract_table(extract_results, sim_row=sim_row)


    # ---------------------------------------------------------------------
    # Event handlers (declared but intentionally empty)
    # ---------------------------------------------------------------------

    def _on_click_run(self):
        # --- 1) Guard: need deviation analysis results ---
        dev = getattr(self.project.current, "deviation_analysis", None)
        if dev is None:
            QMessageBox.warning(
                self,
                "Keine Ähnlichkeitsmatrix vorhanden",
                "Bitte zuerst im Tab 'Abweichungsanalyse' eine Ähnlichkeitsuntersuchung durchführen.",
            )
            return

        sim = getattr(dev, "result_matrix", None)
        if sim is None or not isinstance(sim, np.ndarray) or sim.ndim != 2 or sim.size == 0:
            QMessageBox.warning(
                self,
                "Ungültige Ähnlichkeitsmatrix",
                "Die Ähnlichkeitsmatrix fehlt oder ist leer. Bitte Abweichungsanalyse erneut ausführen.",
            )
            return

        transcript_results = list(getattr(dev, "transcript_preprocessed", None) or [])
        extract_results = list(getattr(dev, "extract_preprocessed", None) or [])

        # --- 2) Map GUI presets -> numeric config (alignment only) ---
        step_v_map = {"schwach": 0.06, "mittel": 0.10, "stark": 0.18}
        step_h_map = {"schwach": 0.08, "mittel": 0.12, "stark": 0.22}
        min_sim_map = {"niedrig": 0.10, "mittel": 0.18, "hoch": 0.28}

        asr_choice = (self.combo_asr_split.currentText() or "mittel").strip().lower()
        tr_choice = (self.combo_tr_merge.currentText() or "mittel").strip().lower()
        min_choice = (self.combo_min_sim.currentText() or "mittel").strip().lower()

        step_v = float(step_v_map.get(asr_choice, 0.10))
        step_h = float(step_h_map.get(tr_choice, 0.12))
        min_sim = float(min_sim_map.get(min_choice, 0.18))

        band = None
        if self.chk_use_band.isChecked():
            b = int(self.spin_band.value())
            band = b if b > 0 else 0

        synch_config = SynchronizationConfig(step_v=step_v, step_h=step_h, min_sim=min_sim, band=band)

        # snapshot extract times (ms) for aligned transcript build
        extract_times_ms = [tm.time for tm in getattr(self.project.asr_extract, "times", [])]

        def work():
            synch_calc = SynchronizationCalculator(config=synch_config)
            synch_result = synch_calc.synchronize(sim)

            col = SynchronizationResultCollection(
                config=synch_config,
                transcript_preprocessed=transcript_results,
                extract_preprocessed=extract_results,
                alignment_path=synch_result.path,
                alignment_ranges_by_transcript=synch_result.ranges_by_transcript,
                total_cost=synch_result.total_cost,
                mean_similarity_on_path=synch_result.mean_similarity_on_path,
                similarity_matrix=sim,
            )
            col.deviation_id = self.project.current.deviation_analysis.result_id
            col.build_aligned_transcript(extract_times_ms)
            return col, synch_result

        def on_success(res):
            col, synch_result = res

            self.project.synchronization_results.append(col)
            self.project.current.synchronization = col
            self.project.synchronization_results.append(col)
            self.project.current.synchronization = col

            self.results = self.project.synchronization_results
            self.sim_matrix = sim
            self.align_path = synch_result.path
            self.align_ranges_by_transcript = synch_result.ranges_by_transcript
            self.current_transcript_index = 0

            self._fill_history_table()
            self._fill_transcript_table(transcript_results)

            if len(transcript_results) > 0:
                self.table_transcript.selectRow(0)
                self._show_transcript_index(0)
            if len(transcript_results) > 0:
                self.table_transcript.selectRow(0)
                self._show_transcript_index(0)

        self.run_in_worker(
            work,
            on_success=on_success,
            busy_widget=self.btn_run,
            status_msg="Synchronisierung läuft…",
        )

    def _on_history_selection_changed(self, selected, deselected):
        rows = self.table_history.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        self._show_history_index(row)


    def _show_history_index(self, row: int):
        if not self.project:
            return

        results = self._history_results
        if row < 0 or row >= len(results):
            return

        col = results[row]

        # update project.current
        if getattr(self.project, "current", None) is not None:
            setattr(self.project.current, "synchronization", col)

        # cache locally
        self.sim_matrix = getattr(col, "similarity_matrix", None)
        self.align_path = getattr(col, "alignment_path", None)
        self.align_ranges_by_transcript = getattr(col, "alignment_ranges_by_transcript", None)
        self.results = results

        transcript_results = getattr(col, "transcript_preprocessed", None) or []
        extract_results = getattr(col, "extract_preprocessed", None) or []

        # refill transcript overview
        self._fill_transcript_table(transcript_results)

        if len(transcript_results) == 0:
            self._clear_table(self.table_extract)
            return

        # reset / clamp current transcript index
        self.current_transcript_index = max(0, min(self.current_transcript_index, len(transcript_results) - 1))

        # select row and show details
        self.table_transcript.selectRow(self.current_transcript_index)
        self._show_transcript_index(self.current_transcript_index)


    def _on_transcript_selection_changed(self, selected, deselected):
        rows = self.table_transcript.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        self._show_transcript_index(row)


    def _show_transcript_index(self, row: int):
        if row < 0:
            return

        # resolve current synchronization result (preferred)
        col = None
        if self.project and getattr(self.project, "current", None) is not None:
            col = getattr(self.project.current, "synchronization", None)

        # fallback to latest
        if col is None and self.project and getattr(self.project, "synchronization_results", None):
            if len(self.project.synchronization_results) > 0:
                col = self.project.synchronization_results[-1]

        # resolve extract list
        extract_results = []
        if col is not None:
            extract_results = getattr(col, "extract_preprocessed", None) or []
        else:
            # worst-case fallback (use local cached, if any)
            extract_results = []

        # cache index
        self.current_transcript_index = int(row)

        # build sim_row
        sim_row = None
        sim = self.sim_matrix
        if sim is None and col is not None:
            sim = getattr(col, "similarity_matrix", None)

        if isinstance(sim, np.ndarray) and sim.ndim == 2:
            if 0 <= self.current_transcript_index < sim.shape[0]:
                sim_row = sim[self.current_transcript_index, :]

        # fill right table
        self._fill_aligned_extract_table(extract_results, sim_row=sim_row)

        # keep UI selection consistent if called via double click / programmatically
        if self.table_transcript.currentRow() != self.current_transcript_index:
            self.table_transcript.selectRow(self.current_transcript_index)