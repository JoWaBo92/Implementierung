import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QAction, QFileDialog,
    QToolBar, QTextEdit, QMessageBox, QVBoxLayout
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import source_document


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oral History Synchronizer")
        self.resize(1000, 650)

        self._create_actions()
        self._create_menubar()
        self._create_toolbar()
        self._create_tabs()
        self.statusBar().showMessage("Bereit")

    def _create_actions(self):
        # Action: Projekt laden
        self.action_load_project = QAction("Projekt öffnen", self)
        self.action_load_project.setShortcut("Ctrl+O")
        self.action_load_project.setStatusTip("Projekt laden")
        self.action_load_project.triggered.connect(self.on_load_project)

        # Action: Transkript laden
        self.action_load_transcript = QAction("Transkript laden", self)
        self.action_load_transcript.setShortcut("Ctrl+T")
        self.action_load_transcript.setStatusTip("Transkript laden")
        self.action_load_transcript.triggered.connect(self.on_load_transcript)

        # Action: ASR-Extrakt laden
        self.action_load_asr_extract = QAction("ASR-Extrakt laden", self)
        self.action_load_asr_extract.setShortcut("Ctrl+R")
        self.action_load_asr_extract.setStatusTip("ASR-Extrakt laden")
        self.action_load_asr_extract.triggered.connect(self.on_load_asr_extract)

        # Action: Speichern
        self.action_save = QAction("Projekt speichern", self)
        self.action_save.setShortcut("Ctrl+S")
        self.action_save.setStatusTip("Projekt speichern")
        #self.action_save.triggered.connect(self.on_load)

        # Action: Exportieren
        self.action_export = QAction("Exportieren", self)
        self.action_export.setShortcut("Ctrl+E")
        self.action_export.setStatusTip("Ergebnis exportieren")

        # Action: Preprocessing
        self.action_preprocessing = QAction("Preprocessing", self)
        self.action_preprocessing.setShortcut("Ctrl+1")

        # Action: Abweichungsuntersuchung
        self.action_deviation = QAction("Abweichungsuntersuchung", self)
        self.action_deviation.setShortcut("Ctrl+2")

        # Action: Synchronisierung
        self.action_alignment = QAction("Synchronisierung", self)
        self.action_alignment.setShortcut("Ctrl+3")

        # Action: Beenden
        self.action_exit = QAction("Beenden", self)
        self.action_exit.setShortcut("Ctrl+Q")
        self.action_exit.setStatusTip("Programm beenden")
        self.action_exit.triggered.connect(self.close)

    def _create_menubar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Datei")
        file_menu.addAction(self.action_load_project)
        file_menu.addAction(self.action_load_transcript)
        file_menu.addAction(self.action_load_asr_extract)
        file_menu.addSeparator()
        file_menu.addAction(self.action_save)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        analysis_menu = menubar.addMenu("Analyse")
        analysis_menu.addAction(self.action_preprocessing)
        analysis_menu.addAction(self.action_deviation)
        analysis_menu.addAction(self.action_alignment)

    def _create_toolbar(self):
        toolbar = QToolBar("Haupt-Toolbar", self)
        toolbar.setMovable(True)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(toolbar)

        toolbar.addAction(self.action_load_project)
        toolbar.addAction(self.action_load_transcript)
        toolbar.addAction(self.action_load_asr_extract)
        toolbar.addSeparator()
        toolbar.addAction(self.action_preprocessing)
        toolbar.addAction(self.action_deviation)
        toolbar.addAction(self.action_alignment)
        toolbar.addSeparator()
        toolbar.addAction(self.action_export)

    def _create_tabs(self):
        self.tabs = QTabWidget()

        self.tab_project = QWidget()
        self.tab_preprocessing = QWidget()
        self.tab_deviation = QWidget()
        self.tab_alignment = QWidget()

        self.editor = QTextEdit()
        self.editor.setPlaceholderText("")

        project_layout = QVBoxLayout(self.tab_project)
        project_layout.addWidget(self.editor)
        self.tab_project.setLayout(project_layout)

        self.tabs.addTab(self.tab_project, "Projekt")
        self.tabs.addTab(self.tab_preprocessing, "Preprocessing")
        self.tabs.addTab(self.tab_deviation, "Abweichungsuntersuchung")
        self.tabs.addTab(self.tab_alignment, "Synchronisierung")

        self.setCentralWidget(self.tabs)

    def _pick_file(self, title: str, filter_str: str = "Alle Dateien (*.*)"):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter_str)
        return path

    def on_load_project(self):
        path = self._pick_file("Projekt laden", "Projektdateien (*.json *.yaml *.yml *.ohs);;Alle Dateien (*.*)")
        if not path:
            return
        self.statusBar().showMessage(f"Projekt geladen: {path}")
        self.editor.setPlainText(f"Projektdatei:\n{path}")

    def on_load_transcript(self):
        path = self._pick_file("Transkript laden", "OpenDocument Text (*.odt);;Alle Dateien (*.*)")
        if not path:
            return

        try:
            transcript = source_document.ManualTranscript(path)
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Transkript konnte nicht geladen werden:\n{e}")
            return

        self.statusBar().showMessage(f"Transkript geladen: {path}")
        self.editor.setPlainText(f"Transkriptdatei:\n{path}\n\n{transcript}")

    def on_load_asr_extract(self):
        path = self._pick_file("ASR-Extrakt laden", "CSV (*.csv);;Alle Dateien (*.*)")
        if not path:
            return
        
        try:
            extract = source_document.ASRExtract(path)
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Transkript konnte nicht geladen werden:\n{e}")
            return
        
        self.statusBar().showMessage(f"ASR-Extrakt geladen: {path}")
        self.editor.setPlainText(f"ASR-Extrakt:\n{path}\n\n{extract}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
