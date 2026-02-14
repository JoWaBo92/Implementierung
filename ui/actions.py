from PyQt5.QtWidgets import QAction


class AppActions:
    """
    Enthält alle Actions der main_window-GUI
    """

    def __init__(self, parent):
        self.new_project = QAction("Neues Projekt", parent)
        self.new_project.setShortcut("Ctrl+N")
        self.new_project.setStatusTip("Neues Projekt erstellen")

        self.load_project = QAction("Projekt öffnen", parent)
        self.load_project.setShortcut("Ctrl+O")
        self.load_project.setStatusTip("Projekt laden")

        self.load_transcript = QAction("Transkript laden", parent)
        self.load_transcript.setShortcut("Ctrl+T")
        self.load_transcript.setStatusTip("Transkript laden")

        self.load_asr_extract = QAction("ASR-Extrakt laden", parent)
        self.load_asr_extract.setShortcut("Ctrl+R")
        self.load_asr_extract.setStatusTip("ASR-Extrakt laden")

        self.save_project = QAction("Projekt speichern", parent)
        self.save_project.setShortcut("Ctrl+S")
        self.save_project.setStatusTip("Projekt speichern")

        self.export = QAction("Exportieren", parent)
        self.export.setShortcut("Ctrl+E")
        self.export.setStatusTip("Ergebnis exportieren")

        self.preprocessing = QAction("Preprocessing", parent)
        self.preprocessing.setShortcut("Ctrl+1")

        self.deviation = QAction("Abweichungsuntersuchung", parent)
        self.deviation.setShortcut("Ctrl+2")

        self.alignment = QAction("Synchronisierung", parent)
        self.alignment.setShortcut("Ctrl+3")

        self.exit = QAction("Beenden", parent)
        self.exit.setShortcut("Ctrl+Q")
        self.exit.setStatusTip("Programm beenden")