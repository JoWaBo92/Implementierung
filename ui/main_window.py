from PyQt5.QtWidgets import QMainWindow, QTabWidget, QWidget, QToolBar, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

from ui.actions import AppActions
from ui.tabs.project_tab import ProjectTab

from domain.source_document import ASRExtract, ManualTranscript


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oral History Synchronizer")
        self.resize(1000, 650)

        self.actions = AppActions(self)
        self._wire_actions()

        self._create_menubar()
        self._create_toolbar()
        self._create_tabs()

        self.statusBar().showMessage("Bereit")

    def _wire_actions(self):
        self.actions.load_project.triggered.connect(self.on_load_project)
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

        self.project_tab = ProjectTab()
        self.tab_preprocessing = QWidget()
        self.tab_deviation = QWidget()
        self.tab_alignment = QWidget()

        self.tabs.addTab(self.project_tab, "Projekt")
        self.tabs.addTab(self.tab_preprocessing, "Preprocessing")
        self.tabs.addTab(self.tab_deviation, "Abweichungsuntersuchung")
        self.tabs.addTab(self.tab_alignment, "Synchronisierung")

        self.setCentralWidget(self.tabs)

    # ---------- Internal helpers ----------
    def _pick_file(self, title: str, filter_str: str = "Alle Dateien (*.*)"):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter_str)
        return path

    def _show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    # ---------- Functions ----------
    def on_load_project(self):
        path = self._pick_file("Projekt laden", "Projektdateien (*.json *.yaml *.yml *.ohs);;Alle Dateien (*.*)")
        if not path:
            return
        self.statusBar().showMessage(f"Projekt geladen: {path}")
        self.project_tab.set_log_text(f"Projektdatei:\n{path}")

    def on_load_transcript(self):
        path = self._pick_file("Transkript laden", "OpenDocument Text (*.odt);;Alle Dateien (*.*)")
        if not path:
            return

        try:
            transcript = ManualTranscript(path)
        except Exception as e:
            self._show_error("Fehler", f"Transkript konnte nicht geladen werden:\n{e}")
            return

        self.statusBar().showMessage(f"Transkript geladen: {path}")
        self.project_tab.set_log_text(f"Transkriptdatei:\n{path}\n\n{transcript}")
        self.project_tab.set_transcript(transcript)

    def on_load_asr_extract(self):
        path = self._pick_file("ASR-Extrakt laden", "CSV (*.csv);;Alle Dateien (*.*)")
        if not path:
            return

        try:
            extract = ASRExtract(path)
        except Exception as e:
            self._show_error("Fehler", f"ASR-Extrakt konnte nicht geladen werden:\n{e}")
            return

        self.statusBar().showMessage(f"ASR-Extrakt geladen: {path}")
        self.project_tab.set_log_text(f"ASR-Extrakt:\n{path}\n\n{extract}")
        self.project_tab.set_asr_extract(extract)
