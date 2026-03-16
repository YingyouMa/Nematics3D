from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QFont



class ScopedConsoleDock(QtWidgets.QDockWidget):

    _sig_append = QtCore.Signal(str) if hasattr(QtCore, "Signal") else QtCore.pyqtSignal(str)
    
    
    def __init__(self, 
                 title:         str="Scoped Console", 
                 parent:        QtWidgets.QMainWindow | None = None,
                 font_size:     float=15):
        
        super().__init__(title, parent)

        self._text = QtWidgets.QPlainTextEdit(self)
        self._text.setReadOnly(True)
        self._text.document().setMaximumBlockCount(20000)  # avoid unbounded growth
        self.setWidget(self._text)
        
        font = QFont("Consolas")
        font.setPointSize(font_size)     
        self._text.setFont(font)

        self._sig_append.connect(self._append_text)
        

    def write(self, msg: str) -> None:
        """Append raw text (no newline added)."""
        if msg:
            self._sig_append.emit(msg)

    def println(self, msg: str) -> None:
        """Append one line (newline added)."""
        self._sig_append.emit((msg or "") + "\n")

    def clear(self) -> None:
        self._text.clear()

    @QtCore.Slot(str) if hasattr(QtCore, "Slot") else QtCore.pyqtSlot(str)
    def _append_text(self, text: str) -> None:
        cursor = self._text.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self._text.setTextCursor(cursor)
        self._text.ensureCursorVisible()
