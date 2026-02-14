from PyQt5.QtWidgets import QMainWindow, QTabWidget, QWidget, QToolBar, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

from persistence.project_repository import save_project, load_project

from ui.actions import AppActions
from ui.tabs.project_tab import ProjectTab
from ui.tabs.preprocessing_tab import PreprocessingTab
from ui.tabs.deviation_tab import DeviationTab
from ui.tabs.synchronization_tab import SynchronizationTab

from domain.source_document import ASRExtract, ManualTranscript
from domain.project import Project


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oral History Synchronizer")
        self.resize(1000, 650)

        self.project = Project()

        self.actions = AppActions(self)
        self._wire_actions()

        self._create_menubar()
        self._create_toolbar()
        self._create_tabs()

        self.statusBar().showMessage("Bereit")
        self.update_ui_state()

    def _wire_actions(self):
        self.actions.new_project.triggered.connect(self.on_new_project)
        self.actions.load_project.triggered.connect(self.on_load_project)
        self.actions.save_project.triggered.connect(self.on_save_project)
        self.actions.load_transcript.triggered.connect(self.on_load_transcript)
        self.actions.load_asr_extract.triggered.connect(self.on_load_asr_extract)
        self.actions.export.triggered.connect(self.on_export)
        self.actions.exit.triggered.connect(self.close)

        # Analyse-Menü: Tab fokussieren
        self.actions.preprocessing.triggered.connect(lambda: self._switch_to_tab(self.tab_preprocessing))
        self.actions.deviation.triggered.connect(lambda: self._switch_to_tab(self.tab_deviation))
        self.actions.alignment.triggered.connect(lambda: self._switch_to_tab(self.tab_alignment))


    def _create_menubar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Datei")
        file_menu.addAction(self.actions.new_project)
        file_menu.addAction(self.actions.load_transcript)
        file_menu.addAction(self.actions.load_asr_extract)
        file_menu.addSeparator()
        file_menu.addAction(self.actions.save_project)
        file_menu.addAction(self.actions.export)
        file_menu.addSeparator()
        file_menu.addAction(self.actions.exit)

        analysis_menu = menubar.addMenu("Analyse")
        analysis_menu.addAction(self.actions.preprocessing)
        analysis_menu.addAction(self.actions.deviation)
        analysis_menu.addAction(self.actions.alignment)

    def _create_toolbar(self):
        toolbar = QToolBar("Haupt-Toolbar", self)
        toolbar.setMovable(True)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(toolbar)

        toolbar.addAction(self.actions.new_project)
        toolbar.addAction(self.actions.load_project)
        toolbar.addAction(self.actions.load_transcript)
        toolbar.addAction(self.actions.load_asr_extract)
        toolbar.addSeparator()
        toolbar.addAction(self.actions.preprocessing)
        toolbar.addAction(self.actions.deviation)
        toolbar.addAction(self.actions.alignment)
        toolbar.addSeparator()
        toolbar.addAction(self.actions.export)

    def _create_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)

        self.tab_project = ProjectTab(project=self.project, parent=self)
        self.tab_preprocessing = PreprocessingTab(project=self.project, parent=self)
        self.tab_deviation = DeviationTab(project=self.project, parent=self)
        self.tab_alignment = SynchronizationTab(project=self.project, parent=self)

        self.tabs.addTab(self.tab_project, "Projekt")
        self.tabs.addTab(self.tab_preprocessing, "Preprocessing")
        self.tabs.addTab(self.tab_deviation, "Abweichungsuntersuchung")
        self.tabs.addTab(self.tab_alignment, "Synchronisierung")

        self.setCentralWidget(self.tabs)

    # ---------- Internal helpers -----------
    def _pick_file(self, title: str, filter_str: str = "Alle Dateien (*.*)"):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter_str)
        return path

    def _show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)


    def _switch_to_tab(self, tab_widget: QWidget):
        try:
            idx = self.tabs.indexOf(tab_widget)
            if idx >= 0:
                self.tabs.setCurrentIndex(idx)
        except Exception:
            pass

    def update_ui_state(self):
        """Aktiviert/Deaktiviert Actions abhängig vom Projektzustand."""
        has_transcript = getattr(self.project, "transcript", None) is not None
        has_asr = getattr(self.project, "asr_extract", None) is not None

        cur = getattr(self.project, "current", None)
        has_pre = bool(cur and getattr(cur, "preprocessing", None))
        has_dev = bool(cur and getattr(cur, "deviation_analysis", None))
        has_sync = bool(cur and getattr(cur, "synchronization", None))

        # Analyse-Actions: nur sinnvoll, wenn die nötigen Inputs existieren
        self.actions.preprocessing.setEnabled(has_transcript and has_asr)
        self.actions.deviation.setEnabled(has_pre)
        self.actions.alignment.setEnabled(has_dev)

        # Export nur nach Synchronisierung
        self.actions.export.setEnabled(has_sync)

    @staticmethod
    def _ms_to_time_str(ms: int) -> str:
        if ms is None or ms < 0:
            return ""
        hh = ms // 3600000
        ms %= 3600000
        mm = ms // 60000
        ms %= 60000
        ss = ms // 1000
        mmm = ms % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{mmm:03d}"

    def on_export(self):
        import csv

        transcript = getattr(self.project, "transcript", None)
        if transcript is None or not getattr(transcript, "segments", None):
            QMessageBox.warning(self, "Export nicht möglich", "Kein Transkript geladen.")
            return

        sync = getattr(getattr(self.project, "current", None), "synchronization", None)
        if sync is None or not getattr(sync, "aligned_transcript", None):
            QMessageBox.warning(
                self,
                "Export nicht möglich",
                "Keine Synchronisierung vorhanden. Bitte zuerst im Tab 'Synchronisierung' ausführen."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export speichern unter",
            "",
            "CSV (*.csv);;Alle Dateien (*.*)"
        )
        if not file_path:
            return

        aligned = sync.aligned_transcript
        segments = transcript.segments
        n = min(len(segments), len(aligned))

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["index", "speaker", "start", "end", "duration_s", "asr_range", "mean_similarity", "text"])

                for i in range(n):
                    seg = segments[i]
                    a = aligned[i]

                    speaker = getattr(seg, "speaker", "")
                    text = getattr(seg, "text", "")

                    start_ms = getattr(a, "start_ms", None)
                    end_ms = getattr(a, "end_ms", None)

                    start_str = self._ms_to_time_str(start_ms) if start_ms is not None else ""
                    end_str = self._ms_to_time_str(end_ms) if end_ms is not None else ""

                    duration_s = ""
                    if start_ms is not None and end_ms is not None and end_ms >= start_ms:
                        duration_s = f"{(end_ms - start_ms) / 1000.0:.3f}"

                    j0 = getattr(a, "extract_j0", -1)
                    j1 = getattr(a, "extract_j1", -1)
                    asr_range = f"{j0}-{j1}" if (isinstance(j0, int) and isinstance(j1, int) and j0 >= 0 and j1 >= 0) else ""

                    mean_sim = getattr(a, "mean_similarity", None)
                    mean_sim_str = f"{float(mean_sim):.4f}" if mean_sim is not None else ""

                    w.writerow([i, speaker, start_str, end_str, duration_s, asr_range, mean_sim_str, text])

            self.statusBar().showMessage(f"Export gespeichert: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export fehlgeschlagen", str(e))

    def on_new_project(self):
        has_sources = bool(getattr(self.project, "transcript", None) or getattr(self.project, "asr_extract", None))
        has_results = bool(hasattr(self.project, "has_analysis_results") and self.project.has_analysis_results())

        if has_sources or has_results:
            res = QMessageBox.warning(
                self,
                "Neues Projekt",
                "Das aktuelle Projekt wird verworfen (inkl. geladener Dateien und Analyse-Ergebnisse).\n\nFortfahren?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if res != QMessageBox.Yes:
                return

        # Neues Projektobjekt
        self.project = Project()

        # Tabs auf neues Projekt setzen
        self.tab_project.set_project(self.project)
        self.tab_preprocessing.set_project(self.project)
        self.tab_deviation.set_project(self.project)
        self.tab_alignment.set_project(self.project)

        # Log / Status
        try:
            self.tab_project.set_log_text("")
        except Exception:
            pass

        self.update_ui_state()
        self.statusBar().showMessage("Neues Projekt erstellt")
        self._switch_to_tab(self.tab_project)

    # ---------- Event Handlers -------------
    def on_load_project(self):
        project_dir = QFileDialog.getExistingDirectory(self, "Projekt öffnen", "", QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if not project_dir:
            return
        
        try:
            self.project = load_project(project_dir)
            pass
        except Exception as e:
            self._show_error("Fehler", f"Projekt konnte nicht geladen werden:\n{e}")
            return

        self.tab_project.set_project(self.project)
        self.tab_preprocessing.set_project(self.project)
        self.tab_deviation.set_project(self.project)
        self.tab_alignment.set_project(self.project)

        self.update_ui_state()
        self.statusBar().showMessage(f"Projekt geladen: {project_dir}")
        self.tab_project.set_log_text(f"Projektdatei:\n{project_dir}")

    def on_save_project(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Projekt speichern unter", "", "Oral History Project (*.ohsproj)")

        if file_path:
            save_project(self.project, file_path)
            print(file_path)

    def on_load_transcript(self):
        path = self._pick_file("Transkript laden", "OpenDocument Text (*.odt);;Alle Dateien (*.*)")
        if not path:
            return
        
        if not self._confirm_reset_analysis_results_if_needed("Transkript"):
            return

        try:
            self.project.transcript = ManualTranscript(path)
        except Exception as e:
            self._show_error("Fehler", f"Transkript konnte nicht geladen werden:\n{e}")
            return

        self.statusBar().showMessage(f"Transkript geladen: {path}")
        self.tab_project.set_log_text(f"Transkript geladen:\n{path}")
        self.tab_project.set_transcript(self.project.transcript)
        self.tab_preprocessing.set_project(self.project)
        self.tab_preprocessing.set_project(self.project)
        self.tab_deviation.set_project(self.project)
        self.tab_alignment.set_project(self.project)
        self.update_ui_state()

    def on_load_asr_extract(self):
        path = self._pick_file("ASR-Extrakt laden", "CSV (*.csv);;Alle Dateien (*.*)")
        if not path:
            return
        
        if not self._confirm_reset_analysis_results_if_needed("ASR-Extrakt"):
            return

        try:
            self.project.asr_extract = ASRExtract(path)
        except Exception as e:
            self._show_error("Fehler", f"ASR-Extrakt konnte nicht geladen werden:\n{e}")
            return

        self.statusBar().showMessage(f"ASR-Extrakt geladen: {path}")
        self.tab_project.set_log_text(f"ASR-Extrakt geladen:\n{path}")
        self.tab_project.set_asr_extract(self.project.asr_extract)
        self.tab_preprocessing.set_project(self.project)
        self.tab_preprocessing.set_project(self.project)
        self.tab_deviation.set_project(self.project)
        self.tab_alignment.set_project(self.project)
        self.update_ui_state()

    def on_tab_changed(self, index: int):
        widget = self.tabs.widget(index)

        if hasattr(widget, "on_tab_activated"):
            widget.on_tab_activated()
        self.update_ui_state()

    def _show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    def _confirm_reset_analysis_results_if_needed(self, what: str) -> bool:
        if not hasattr(self.project, "has_analysis_results") or not self.project.has_analysis_results():
            return True

        msg = (
            f"Wenn ein neues {what} geladen wird, werden alle bisherigen Ergebnisse aus\n"
            f"Preprocessing, Abweichungsuntersuchung und Synchronisierung gelöscht.\n\n"
            f"Fortfahren?"
        )
        res = QMessageBox.warning(
            self,
            "Vorhandene Ergebnisse werden gelöscht",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if res != QMessageBox.Yes:
            return False

        # wirklich löschen
        if hasattr(self.project, "clear_analysis_results"):
            self.project.clear_analysis_results()
        return True