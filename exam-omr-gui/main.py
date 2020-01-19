#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py

import os.path
import sys
import json
import csv

import cv2 as cv
import numpy as np
import fitz
from PySide2.QtCore import Slot, Signal, Qt, QSettings
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog, QListWidgetItem, QAction

from ui_mainwindow import Ui_MainWindow
from ui_saveresult import Ui_SaveResultDialog
from ui_setversion import Ui_SetVersionDialog
from exam_omr import process_answer_sheet, get_testset_answers


def cv_to_qpixmap(cvimg):
    qformat = QImage.Format_Indexed8
    if len(cvimg.shape) == 3:
        if cvimg.shape[2] == 4:
            qformat = QImage.Format_RGBA8888
        else:
            qformat = QImage.Format_RGB888
    qimg = QImage(cvimg.data,
                  cvimg.shape[1],
                  cvimg.shape[0],
                  cvimg.strides[0],
                  qformat)
    qimg = qimg.rgbSwapped()

    return QPixmap.fromImage(qimg)


def fitz_to_cv(doc, ref):
    return cv.imdecode(np.frombuffer(doc.extractImage(ref)['image'], dtype=np.uint8),
                       cv.IMREAD_COLOR)

class SaveResultDialog(QDialog):
    def __init__(self, records):
        super(SaveResultDialog, self).__init__()
        self.ui = Ui_SaveResultDialog()
        self.ui.setupUi(self)
        for row in records:
            self.ui.comboBox.addItem(row[1])
        self.ui.comboBox.lineEdit().selectAll()

    def accept(self):
        self.done(self.ui.comboBox.currentIndex())

    def reject(self):
        self.done(-1)

    @staticmethod
    def getRecordId(records):
        dialog = SaveResultDialog(records)
        dialog.exec_()
        dialog.show()
        return dialog.result()


class SetVersionDialog(QDialog):
    def __init__(self):
        super(SetVersionDialog, self).__init__()
        self.ui = Ui_SetVersionDialog()
        self.ui.setupUi(self)

    def accept(self):
        version = 0
        try:
            version = int(self.ui.versionEdit.text())
        except ValueError:
            pass
        self.done(version)

    def reject(self):
        self.done(0)

    @staticmethod
    def getManualVersion():
        dialog = SetVersionDialog()
        dialog.exec_()
        dialog.show()
        return dialog.result()


class MainWindow(QMainWindow):
    img = None
    processed_img = None
    pdf = None
    pdf_image_refs = []
    pdf_image_idx = -1
    pixmap = None
    qr_code_id = None
    variant = 0
    valid_answers = {}
    score = -1.0
    updateWidgetsState = Signal()
    updateImageView = Signal()
    updateSheetResults = Signal()
    settings = None
    template_filename = None
    grading_script_filename = None
    saved_grading_script_filename = None
    student_records_filename = None
    saved_student_records_filename = None
    student_records = []
    question_variants = []
    answers_variants = []
    answer_scores = []
    recent_files = []
    recent_files_actions = []
    max_recent_files = 5

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.action_Open_image.triggered.connect(self.openImage)
        self.ui.action_Open_PDF_collection.triggered.connect(self.openPDF)
        self.ui.action_Set_template_file.triggered.connect(self.setTemplateFile)
        self.ui.action_Set_grading_R_file.triggered.connect(self.setGradingScriptFile)
        self.ui.action_Set_student_records_file.triggered.connect(self.setStudentRecordsFile)
        self.ui.nextButton.clicked.connect(self.nextPdfImage)
        self.ui.prevButton.clicked.connect(self.prevPdfImage)
        self.updateWidgetsState.connect(self.onUpdateWidgetsState)
        self.updateImageView.connect(self.onUpdateImageView)
        self.updateSheetResults.connect(self.onUpdateSheetResults)
        self.ui.processButton.clicked.connect(self.processImage)
        self.ui.saveResultButton.clicked.connect(self.saveResult)
        self.settings = QSettings()
        if self.settings.contains("template_file"):
            self.template_filename = self.settings.value("template_file")
        if self.settings.contains("student_records_file"):
            self.saved_student_records_filename = self.settings.value("student_records_file")
        if self.settings.contains("grading_script_file"):
            self.saved_grading_script_filename = self.settings.value("grading_script_file")
        if self.settings.contains("recent_files"):
            try:
                self.recent_files = json.loads(self.settings.value("recent_files"))
            except (ValueError, TypeError):
                self.recent_files = []
        self.updateWidgetsState.emit()

    def resizeEvent(self, event):
        self.updateImageView.emit()
        return super(MainWindow, self).resizeEvent(event)

    def resetSheetResults(self):
        self.processed_img = None
        self.pixmap = None
        self.qr_code_id = None
        self.variant = 0
        self.valid_answers = {}
        self.score = -1

    def updatePdfSelection(self):
        self.resetSheetResults()
        self.img = fitz_to_cv(self.pdf, self.pdf_image_refs[self.pdf_image_idx])
        self.pixmap = cv_to_qpixmap(cv.cvtColor(self.img, cv.COLOR_BGR2RGB))
        self.updateWidgetsState.emit()
        self.updateImageView.emit()
        self.updateSheetResults.emit()

    def updateRecentFiles(self, filename):
        try:
            self.recent_files.remove(filename)
        except ValueError:
            pass
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[1:]
        self.recent_files.append(filename)
        self.settings.setValue('recent_files', json.dumps(self.recent_files))

    def loadImage(self, filename):
        if os.path.isfile(filename):
            self.img = cv.imread(filename)
            self.updateRecentFiles(filename)
            self.pdf = None
            self.pdf_image_refs = []
            self.pdf_image_idx = -1
            self.resetSheetResults()
            self.updateWidgetsState.emit()
            self.pixmap = cv_to_qpixmap(cv.cvtColor(self.img, cv.COLOR_BGR2RGB))
            self.updateImageView.emit()
            self.updateSheetResults.emit()

    def loadPDF(self, filename):
        if os.path.isfile(filename):
            self.pdf = fitz.open(filename)
            self.updateRecentFiles(filename)
            self.pdf_image_refs = []
            for p in range(len(self.pdf)):
                self.pdf_image_refs += [obj[0] for obj in self.pdf.getPageImageList(p)]
            self.pdf_image_idx = 0
            self.updatePdfSelection()

    def computeScore(self, variant, answers, question_variants, answers_variants, answer_scores):
        choice_to_int = {"a": 0, "b": 1, "c": 2, "d": 3}
        score = 0.0
        for q, a in answers.items():
            orig_q = question_variants[variant - 1].index(q)
            orig_a = answers_variants[variant - 1][orig_q][choice_to_int[a.lower()]]-1
            score += answer_scores[orig_q][orig_a]
            if answer_scores[orig_q][orig_a] < 0.1:
                print("Wrong answer for question {:d} in variant {:d}".format(q, variant))
        return score

    @Slot()
    def openImage(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        self.loadImage(filename)

    @Slot()
    def openPDF(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Open PDF collection", "", "PDF Files (*.pdf)")
        self.loadPDF(filename)

    @Slot()
    def openRecent(self):
        action = self.sender()

        if action:
            filename = action.data()
            _, extension = os.path.splitext(filename)
            if extension.lower() == ".pdf":
                self.loadPDF(filename)
            else:
                self.loadImage(filename)

    @Slot()
    def setTemplateFile(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Set answer sheet template file", "", "SVG Template Files (*.svg)")
        if os.path.isfile(filename):
            self.template_filename = filename
            self.settings.setValue("template_file", self.template_filename)
            self.updateWidgetsState.emit()

    @Slot()
    def setGradingScriptFile(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Set R grading file",
                                                  self.saved_grading_script_filename,
                                                  "R Script Files (*.r)")
        if os.path.isfile(filename):
            self.grading_script_filename = filename
            self.saved_grading_script_filename = filename
            self.settings.setValue("grading_script_file", self.grading_script_filename)

    @Slot()
    def setStudentRecordsFile(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Set student records file",
                                                  self.saved_student_records_filename,
                                                  "CSV Files (*.csv)")
        if os.path.isfile(filename):
            self.student_records_filename = filename
            self.saved_student_records_filename = filename
            self.settings.setValue("student_records_file", self.student_records_filename)
            with open(self.student_records_filename) as csvfile:
                records = csv.reader(csvfile, delimiter=",", quotechar='"')
                self.student_records = []
                for row in records:
                    while len(row) < 4:
                        row.append("")
                    if row[2] == "":
                        row[2] = "0"
                    if row[3] == "":
                        row[3] = "-1"
                    self.student_records.append(row)
                csvfile.close()
            self.updateWidgetsState.emit()

    @Slot()
    def nextPdfImage(self):
        self.pdf_image_idx += 1
        self.updatePdfSelection()

    @Slot()
    def prevPdfImage(self):
        self.pdf_image_idx -= 1
        self.updatePdfSelection()

    @Slot()
    def processImage(self):
        if self.img is not None:
            if self.grading_script_filename is None:
                self.setGradingScriptFile()
            if os.path.isfile(self.grading_script_filename):
                self.question_variants, \
                self.answers_variants, \
                self.answer_scores = get_testset_answers(self.grading_script_filename)
            self.processed_img, \
            self.qr_code_id, \
            self.variant, \
            self.valid_answers = process_answer_sheet(self.img, self.template_filename)
            self.pixmap = cv_to_qpixmap(cv.cvtColor(self.processed_img, cv.COLOR_BGR2RGB))
            if self.variant == 0:
                self.variant = SetVersionDialog.getManualVersion()
            if 0 < self.variant and len(self.question_variants) > 0 and \
                    len(self.answers_variants) > 0 and len(self.answer_scores) > 0:
                self.score = self.computeScore(self.variant,
                                               self.valid_answers,
                                               self.question_variants,
                                               self.answers_variants,
                                               self.answer_scores)
            self.updateWidgetsState.emit()
            self.updateImageView.emit()
            self.updateSheetResults.emit()

    @Slot()
    def saveResult(self):
        if self.score > -1.0:
            if self.student_records_filename is None or len(self.student_records) == 0:
                self.setStudentRecordsFile()
            if len(self.student_records) > 0:
                stud_id = SaveResultDialog.getRecordId(self.student_records)
                if 0 <= stud_id < len(self.student_records):
                    self.student_records[stud_id][2] = "{:.0f}".format(self.score)
                    if self.pdf_image_idx >= 0 and len(self.pdf_image_refs) > 0:
                        self.student_records[stud_id][3] = "{:d}".format(self.pdf_image_idx)
                        self.updateWidgetsState.emit()
                    with open(self.student_records_filename, "w") as csvfile:
                        records = csv.writer(csvfile, delimiter=",", quotechar='"')
                        records.writerows(self.student_records)
                        csvfile.close()

    @Slot()
    def onUpdateWidgetsState(self):
        self.ui.action_Open_PDF_collection.setEnabled(self.template_filename is not None)
        pdf_whats_this = "Open PDF collection of scanned answer sheets" if self.template_filename is not None \
            else "Must set template file"
        self.ui.action_Open_PDF_collection.setWhatsThis(pdf_whats_this)
        self.ui.action_Open_image.setEnabled(self.template_filename is not None)
        self.ui.menu_Open_recent.setEnabled(self.template_filename is not None and len(self.recent_files) > 0)
        self.ui.menu_Open_recent.clear()
        self.recent_files_actions = []
        self.recent_files = [f for f in self.recent_files if os.path.isfile(f)]
        for idx, filename in enumerate(self.recent_files[::-1]):
            action = QAction("&{:d} {:s}".format(idx + 1, os.path.basename(filename)))
            action.setToolTip(filename)
            action.setVisible(True)
            action.setData(filename)
            action.triggered.connect(self.openRecent)
            self.recent_files_actions.append(action)
            self.ui.menu_Open_recent.addAction(action)
        self.ui.nextButton.setEnabled(0 <= self.pdf_image_idx < len(self.pdf_image_refs) - 1)
        self.ui.prevButton.setEnabled(self.pdf_image_idx > 0)
        self.ui.processButton.setEnabled(self.img is not None)
        self.ui.saveResultButton.setEnabled(len(self.valid_answers) > 0)
        self.ui.statusbar.clearMessage()
        status_message = ""
        if self.pdf_image_idx >= 0 and len(self.pdf_image_refs) > 0:
            score_message = ", no records file set"
            if self.student_records_filename is not None and len(self.student_records) > 0:
                score_message = ", not found in records"
                for record in self.student_records:
                    try:
                        stud_id = int(record[3])
                        stud_score = float(record[2])
                        stud_name = record[1]
                        stud_group = record[0]
                        if stud_id == self.pdf_image_idx:
                            score_message = ", in records as '{:s}' from {:s} "\
                                            "with a score of {:.0f}".format(stud_name, stud_group, stud_score)
                            break
                    except ValueError:
                        pass
            status_message = "Answer sheet {:2d} of {:d}{:s}".format(self.pdf_image_idx + 1,
                                                                     len(self.pdf_image_refs),
                                                                     score_message)
        self.ui.statusbar.showMessage(status_message)

    @Slot()
    def onUpdateImageView(self):
        if self.pixmap is not None:
            self.ui.imageView.setPixmap(self.pixmap.scaled(self.ui.imageView.size(),
                                                           Qt.KeepAspectRatio,
                                                           Qt.SmoothTransformation))

    @Slot()
    def onUpdateSheetResults(self):
        self.ui.idText.setText(self.qr_code_id)
        self.ui.versionText.setText('{:d}'.format(self.variant) if self.variant > 0 else '')
        scoreText = "{:.0f}".format(self.score) if self.score >= 0.0 else 'Not scored yet.' if self.variant > 0 else ''
        self.ui.scoreText.setText(scoreText)
        self.ui.answerList.clear()
        for q in sorted(self.valid_answers, key=lambda x: int(x)):
            self.ui.answerList.addItem(QListWidgetItem("{:3d} -> {:s}".format(int(q), self.valid_answers[q].upper())))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setOrganizationDomain("tuiasi.ro")
    app.setOrganizationName('"Gheorghe Asachi" Technical University of Iași')
    app.setApplicationName("Answer sheet OMR")
    app.setApplicationVersion("0.1")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
