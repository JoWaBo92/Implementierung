from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool, Qt
from PyQt5.QtWidgets import QHeaderView, QSizePolicy

class TableUtils:
    """Helpers to keep QTableWidget configuration consistent and fast."""

    @staticmethod
    def configure_table_basic(table, *, selection_rows=True, alternating=True):
        from PyQt5.QtWidgets import QAbstractItemView
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows if selection_rows else QAbstractItemView.SelectItems)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(bool(alternating))
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setHighlightSections(False)

    @staticmethod
    def configure_index_text_table(table, *, index_cols=(0,), stretch_col=-1):
        from PyQt5.QtWidgets import QHeaderView
        header = table.horizontalHeader()
        for c in index_cols:
            if c >= 0:
                header.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        if stretch_col >= 0:
            header.setSectionResizeMode(stretch_col, QHeaderView.Stretch)
        header.setStretchLastSection(False)

    @staticmethod
    def configure_all_interactive(table):
        from PyQt5.QtWidgets import QHeaderView
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def configure_table_fill(table, text_cols=None, resize_cols=None):
        """
        Konfiguriert eine QTableWidget so, dass sie den verfügbaren Platz optimal nutzt.

        text_cols   → Spalten, die den restlichen Platz strecken sollen
        resize_cols → Spalten, die automatisch auf Inhalt angepasst werden
        """
        text_cols = text_cols or []
        resize_cols = resize_cols or []

        # Tabelle soll wachsen dürfen
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Lesbarkeit
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setTextElideMode(Qt.ElideRight)

        hh = table.horizontalHeader()
        hh.setCascadingSectionResizes(False)
        hh.setStretchLastSection(True)

        # Default: Interactive
        hh.setSectionResizeMode(QHeaderView.Interactive)

        # Kurze Spalten automatisch an Inhalt anpassen
        for col in resize_cols:
            hh.setSectionResizeMode(col, QHeaderView.ResizeToContents)

        # Textspalten strecken
        for col in text_cols:
            hh.setSectionResizeMode(col, QHeaderView.Stretch)

    


class WorkerSignals(QObject):
    finished = pyqtSignal(object)   # result
    error = pyqtSignal(Exception)


class Worker(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    def run(self):
        try:
            res = self.fn()
            self.signals.finished.emit(res)
        except Exception as e:
            self.signals.error.emit(e)


class BaseTab:
    """Mixin-like base providing worker helper; tabs may inherit QWidget first."""
    def _init_base_tab(self, project, parent=None):
        self.project = project
        self._thread_pool = QThreadPool.globalInstance()

    def set_project(self, project):
        self.project = project

    def on_tab_activated(self):
        pass

    def run_in_worker(self, work_fn, *, on_success, on_error=None, busy_widget=None, status_msg=None):
        from PyQt5.QtWidgets import QApplication, QMessageBox
        if busy_widget is not None:
            busy_widget.setEnabled(False)

        if status_msg and hasattr(self.parent(), "statusBar"):
            self.parent().statusBar().showMessage(status_msg)

        QApplication.setOverrideCursor(Qt.WaitCursor)

        w = Worker(work_fn)

        def _done(result):
            try:
                on_success(result)
            finally:
                if busy_widget is not None:
                    busy_widget.setEnabled(True)
                QApplication.restoreOverrideCursor()
                win = self.window()
                if hasattr(win, "update_ui_state"):
                    win.update_ui_state()

        def _err(exc: Exception):
            try:
                if on_error:
                    on_error(exc)
                else:
                    QMessageBox.critical(self, "Fehler", str(exc))
            finally:
                if busy_widget is not None:
                    busy_widget.setEnabled(True)
                QApplication.restoreOverrideCursor()

        w.signals.finished.connect(_done)
        w.signals.error.connect(_err)
        self._thread_pool.start(w)

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