# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'setversion.ui',
# licensing of 'setversion.ui' applies.
#
# Created: Thu Jan 24 23:15:44 2019
#      by: pyside2-uic  running on PySide2 5.12.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_SetVersionDialog(object):
    def setupUi(self, SetVersionDialog):
        SetVersionDialog.setObjectName("SetVersionDialog")
        SetVersionDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        SetVersionDialog.resize(403, 170)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SetVersionDialog.sizePolicy().hasHeightForWidth())
        SetVersionDialog.setSizePolicy(sizePolicy)
        SetVersionDialog.setModal(True)
        self.verticalLayout = QtWidgets.QVBoxLayout(SetVersionDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(SetVersionDialog)
        self.label_2.setMinimumSize(QtCore.QSize(0, 50))
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.label = QtWidgets.QLabel(SetVersionDialog)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.versionEdit = QtWidgets.QLineEdit(SetVersionDialog)
        self.versionEdit.setInputMask("")
        self.versionEdit.setMaxLength(2)
        self.versionEdit.setObjectName("versionEdit")
        self.verticalLayout.addWidget(self.versionEdit)
        self.buttonBox = QtWidgets.QDialogButtonBox(SetVersionDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(SetVersionDialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), SetVersionDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), SetVersionDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(SetVersionDialog)

    def retranslateUi(self, SetVersionDialog):
        SetVersionDialog.setWindowTitle(QtWidgets.QApplication.translate("SetVersionDialog", "Warning!", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("SetVersionDialog", "No version number could be detected, please set it manually.", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("SetVersionDialog", "Version:", None, -1))

