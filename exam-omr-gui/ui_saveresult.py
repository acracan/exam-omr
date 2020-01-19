# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'saveresult.ui',
# licensing of 'saveresult.ui' applies.
#
# Created: Thu Jan 24 23:15:48 2019
#      by: pyside2-uic  running on PySide2 5.12.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_SaveResultDialog(object):
    def setupUi(self, SaveResultDialog):
        SaveResultDialog.setObjectName("SaveResultDialog")
        SaveResultDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        SaveResultDialog.resize(400, 114)
        SaveResultDialog.setModal(True)
        self.verticalLayout = QtWidgets.QVBoxLayout(SaveResultDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(SaveResultDialog)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.comboBox = QtWidgets.QComboBox(SaveResultDialog)
        self.comboBox.setEditable(True)
        self.comboBox.setObjectName("comboBox")
        self.verticalLayout.addWidget(self.comboBox)
        self.buttonBox = QtWidgets.QDialogButtonBox(SaveResultDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(SaveResultDialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), SaveResultDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), SaveResultDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(SaveResultDialog)

    def retranslateUi(self, SaveResultDialog):
        SaveResultDialog.setWindowTitle(QtWidgets.QApplication.translate("SaveResultDialog", "Save result", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("SaveResultDialog", "Type student\'s name:", None, -1))

