from PyQt5.QtWidgets import QMainWindow, QTabWidget, QWidget, QToolBar, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

from persistence.project_repository import save_project, load_project

from ui.actions import AppActions
from ui.tabs.project_tab import ProjectTab
from ui.tabs.preprocessing_tab import PreprocessingTab
from ui.tabs.deviation_tab import DeviationTab

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

    def _wire_actions(self):
        self.actions.load_project.triggered.connect(self.on_load_project)
        self.actions.save_project.triggered.connect(self.on_save_project)
        self.actions.load_transcript.triggered.connect(self.on_load_transcript)
        self.actions.load_asr_extract.triggered.connect(self.on_load_asr_extract)
        self.actions.exit.triggered.connect(self.close)

    def _create_menubar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Datei")
        file_menu.addAction(self.actions.load_project)
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
        self.tab_alignment = QWidget()

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

        try:
            self.project.transcript = ManualTranscript(path)
        except Exception as e:
            self._show_error("Fehler", f"Transkript konnte nicht geladen werden:\n{e}")
            return

        self.statusBar().showMessage(f"Transkript geladen: {path}")
        self.tab_project.set_log_text(f"Transkript geladen:\n{path}")
        self.tab_project.set_transcript(self.project.transcript)

    def on_load_asr_extract(self):
        path = self._pick_file("ASR-Extrakt laden", "CSV (*.csv);;Alle Dateien (*.*)")
        if not path:
            return

        try:
            self.project.asr_extract = ASRExtract(path)
        except Exception as e:
            self._show_error("Fehler", f"ASR-Extrakt konnte nicht geladen werden:\n{e}")
            return

        self.statusBar().showMessage(f"ASR-Extrakt geladen: {path}")
        self.tab_project.set_log_text(f"ASR-Extrakt geladen:\n{path}")
        self.tab_project.set_asr_extract(self.project.asr_extract)

    def on_tab_changed(self, index: int):
        widget = self.tabs.widget(index)

        if hasattr(widget, "on_tab_activated"):
            widget.on_tab_activated()