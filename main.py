# -*- coding: utf-8 -*-
import os
import sys
import time     # os, sys, time：Python 标准库，用于文件路径、系统参数、计时
import cv2      # cv2：OpenCV，用于图像读取、处理、显示。
import numpy as np      # 数组操作，OpenCV 图像本质是 numpy 数组。
from PIL import ImageFont       # PIL 库，用于在图像上绘制中文文字（因为 OpenCV 原生不支持中文）。
from PyQt5.QtCore import ( # 导入 PyQt5 的核心模块
    Qt,            # 包含一些常量，比如 AlignCenter。
    QTimer,        # 定时器，用于定时执行任务（比如视频逐帧处理
    QThread,       # 多线程，防止界面卡顿（视频保存用）
    pyqtSignal,    # 线程间通信信号
    QCoreApplication # 提供退出应用程序的方法 quit()
)
from PyQt5.QtWidgets import ( # 导入 PyQt5 的各种控件
    QApplication,  # 每个 GUI 程序必须有一个。管理事件循环和全局设置
    QMainWindow,   # 主窗口（你的软件主界面）
    QFileDialog,   # 打开文件/文件夹对话框
    QMessageBox,   # 消息框（提示、确认）
    QWidget,       # 所有可视控件的基类（这里可能未直接使用，但被其他类继承）。
    QHeaderView,   # 表格头部视图，用来设置列宽等
    QTableWidgetItem, # 表格中的一个单元格项
    QAbstractItemView # 提供了表格的选择模式（如 SelectRows）
)

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Qt Designer生成的UI导入
sys.path.append(os.path.join(BASE_DIR, 'UIProgram'))              # 把 UIProgram 文件夹加入模块搜索路径
from UIProgram.UiMain import Ui_MainWindow  # 导入由 .ui 文件生成的界面类
from UIProgram.QssLoader import QSSLoader   # 一个自定义的样式表加载器，用于读取 .css 文件并应用到窗口。
from UIProgram.precess_bar import ProgressBar  # 导入进度条控件（用于视频保存进度）
import detect_tools as tools               # 导入自己定义的工具函数（绘图、坐标转换等）
import Config                              # 导入配置（保存路径、模型路径等）

# 定义主窗口类
class DetectionApp(QMainWindow): #  继承自 QMainWindow
    def __init__(self, parent=None): # 构造函数 
        super().__init__(parent) # 调用父类 QMainWindow 的构造函数，完成窗口的基本初始化
        self.ui = Ui_MainWindow() # 创建界面对象（由 .ui 生成的类）
        self.ui.setupUi(self)  # 将界面上所有的控件（按钮、标签、表格等）创建出来并布置到当前窗口 self 上
                               # 之后我们就可以通过 self.ui.控件对象名 来访问它们

        # 初始化应用 调用三个自定义初始化方法
        self._setup_ui() #设置一些界面的额外属性
        self._connect_signals() # 把按钮点击等事件连接到具体的函数
        self._load_stylesheet() # 加载 CSS 美化界面

        # 模型和资源初始化
        self._init_detection_resources() # 加载 YOLO 模型、字体

    def _setup_ui(self):
        """初始化UI界面设置"""
        self.display_width = 700
        self.display_height = 500 # 用来限定界面上显示图像的区域大小
        self.source_path = None # 记录当前打开的图片/视频/文件夹的路径
        self.camera_active = False # 标志摄像头是否已开启
        self.video_capture = None # OpenCV 的视频捕获对象，用于视频或摄像头

        # 配置表格控件
        table = self.ui.tableWidget  # 获取界面上的表格控件
        table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed) # 设置垂直表头（行号区域）的调整模式为固定，不允许用户拖动改变行高。
        table.verticalHeader().setDefaultSectionSize(40) # 设置每行的默认高度为 40 像素
        table.setColumnWidth(0, 80)  # ID列 设置第 0 列宽度为 80 像素，以此类推
        table.setColumnWidth(1, 200)  # 路径列
        table.setColumnWidth(2, 150)  # 类别列
        table.setColumnWidth(3, 90)  # 置信度列
        table.setColumnWidth(4, 230)  # 位置列
        table.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置表格的选择行为为选中整行
        table.verticalHeader().setVisible(False) # 隐藏垂直表头（即不显示行号那一列）
        table.setAlternatingRowColors(True) # 使表格的行交替使用不同颜色，便于阅读

    def _connect_signals(self): # 在 PyQt5 中，控件的信号（如按钮被点击）通过 .connect() 方法连接到槽函数（即事件处理函数）。
        """连接按钮信号与槽函数"""
        self.ui.PicBtn.clicked.connect(self._handle_image_input) # 界面上的“图片”按钮
        self.ui.comboBox.activated.connect(self._update_selection) # 下拉选择框 （选择已被检测到的目标）
        self.ui.VideoBtn.clicked.connect(self._handle_video_input) # “视频”按钮
        self.ui.CapBtn.clicked.connect(self._toggle_camera) # “摄像头”按钮
        self.ui.SaveBtn.clicked.connect(self._save_results) # “保存结果”按钮
        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit) # “退出”按钮，直接调用 Qt 提供的退出方法
        self.ui.FilesBtn.clicked.connect(self._process_image_batch) # “批量检测图片”按钮

    def _load_stylesheet(self):
        """加载CSS样式表"""
        style_file = os.path.join(BASE_DIR, 'UIProgram', 'style.css')
        qss = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qss)

    def _init_detection_resources(self):
        """初始化检测相关资源"""
        # 加载YOLOv8模型
        self.detector = YOLO(os.path.join(BASE_DIR, 'runs', 'detect', 'exp5', 'weights', 'best.pt'), task='detect')
        self.detector(np.zeros((48, 48, 3)))  # 用一张 48x48 的黑色图片做一次推理（预热）

        # 初始化字体和颜色
        self.detection_font = ImageFont.truetype(os.path.join(BASE_DIR, "Font", "platech.ttf"), 25, 0)
        self.color_palette = tools.Colors() 

        # 初始化定时器
        self.frame_timer = QTimer() # 创建一个 Qt 定时器，用于视频/摄像头的逐帧处理
        self.frame_timer.timeout.connect(self._process_video_frame)
        self.save_timer = QTimer() # 这个未使用，可能是预留

    def _handle_image_input(self):
        """处理单张图片输入"""
        self._stop_video_capture()

        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', './', "图片文件 (*.jpg *.jpeg *.png)")
        if not file_path:
            return

        self._process_single_image(file_path)

    def _process_single_image(self, image_path):
        """处理并显示单张图片的检测结果"""
        self.source_path = image_path
        self.ui.comboBox.setEnabled(True)

        # 读取并检测图片
        start_time = time.time()
        detection_results = self.detector(image_path)[0]
        processing_time = time.time() - start_time

        # 解析检测结果
        boxes = detection_results.boxes.xyxy.tolist() # detection_results.boxes 是一个 Boxes 对象，它封装了检测到的所有边界框的信息。
        # .xyxy 是一个属性，返回一个 torch.Tensor，形状为 (N, 4)，
        # 其中 N 是检测到的目标数量。每一行为 [x1, y1, x2, y2]，坐标是绝对像素坐标（相对于原始图像尺寸），左上角 (x1, y1)，右下角 (x2, y2)。   
        self.detection_boxes = [list(map(int, box)) for box in boxes]
        # 最终 self.detection_boxes 是一个列表，其中每个元素是一个包含四个整数的列表，表示一个检测框的整数坐标。
        # 用于后续在界面上显示坐标、绘制框时直接使用。
        self.detection_classes = detection_results.boxes.cls.int().tolist()
        # .cls 也是一个张量，形状 (N,)，存储每个检测目标的类别索引（整数）。
        # 对于裂缝检测，通常是 0（因为只有一类）
        confidences = detection_results.boxes.conf.tolist()
        # .conf 是形状为 (N,) 的张量，存储每个检测目标的置信度（概率），范围 0~1。
        self.confidence_scores = [f'{score * 100:.2f}%' for score in confidences]

        # 更新UI显示
        self._update_detection_display(detection_results, processing_time) # 将检测得到的带框图像缩放并显示到界面的 label_show 上，
                                                                           # 同时显示耗时、检测数量、图片路径等
        self._update_object_selection()  # 根据检测到的目标数量，重新填充下拉选择框
        self._show_detection_details()   # 默认显示第一个目标的详细信息（类别、置信度、坐标
        self._display_results_table(image_path) # 将所有的检测结果（序号、路径、类别、置信度、坐标）填入表格控件中

    def _update_detection_display(self, results, process_time):
        """更新检测结果显示"""
        # 显示处理时间
        self.ui.time_lb.setText(f'{process_time:.3f} s')

        # 获取带标注的图像
        annotated_img = results.plot()
        self.current_result = annotated_img

        # 调整并显示图像
        width, height = self._calculate_display_size(annotated_img)
        resized_img = cv2.resize(annotated_img, (width, height))
        qimage = tools.cvimg_to_qpiximg(resized_img)

        self.ui.label_show.setPixmap(qimage)
        self.ui.label_show.setAlignment(Qt.AlignCenter)
        self.ui.PiclineEdit.setText(self.source_path)

        # 更新检测数量
        self.ui.label_nums.setText(str(len(self.detection_classes)))

    def _calculate_display_size(self, image):
        """计算适合显示的图像尺寸"""
        img_height, img_width = image.shape[:2]
        aspect_ratio = img_width / img_height

        if aspect_ratio >= self.display_width / self.display_height:
            width = self.display_width
            height = int(width / aspect_ratio)
        else:
            height = self.display_height
            width = int(height * aspect_ratio)

        return width, height

    def _update_object_selection(self):
        """更新目标选择下拉框"""
        options = ['全部']
        target_labels = [
            f'{Config.names[cls_id]}_{idx}'
            for idx, cls_id in enumerate(self.detection_classes)
        ]
        options.extend(target_labels)

        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(options)

    def _show_detection_details(self, index=0):
        """显示检测目标的详细信息"""
        if not self.detection_boxes:
            self._clear_detection_details()
            return

        box = self.detection_boxes[index]
        self.ui.type_lb.setText(Config.CH_names[self.detection_classes[index]])
        self.ui.label_conf.setText(self.confidence_scores[index])
        self.ui.label_xmin.setText(str(box[0]))
        self.ui.label_ymin.setText(str(box[1]))
        self.ui.label_xmax.setText(str(box[2]))
        self.ui.label_ymax.setText(str(box[3]))

    def _clear_detection_details(self):
        """清空检测详情显示"""
        self.ui.type_lb.setText('')
        self.ui.label_conf.setText('')
        self.ui.label_xmin.setText('')
        self.ui.label_ymin.setText('')
        self.ui.label_xmax.setText('')
        self.ui.label_ymax.setText('')

    def _display_results_table(self, source_path):
        """在表格中显示检测结果"""
        table = self.ui.tableWidget
        table.setRowCount(0)
        table.clearContents()

        for idx, (box, cls_id, conf) in enumerate(zip(
                self.detection_boxes, self.detection_classes, self.confidence_scores)):

            row = table.rowCount()
            table.insertRow(row)

            # 添加表格项
            items = [
                QTableWidgetItem(str(row + 1)),  # ID
                QTableWidgetItem(source_path),  # 路径
                QTableWidgetItem(Config.CH_names[cls_id]),  # 类别
                QTableWidgetItem(conf),  # 置信度
                QTableWidgetItem(str(box))  # 位置坐标
            ]

            # 设置文本居中
            for item in [items[0], items[2], items[3]]:
                item.setTextAlignment(Qt.AlignCenter)

            # 添加到表格
            for col, item in enumerate(items):
                table.setItem(row, col, item)

        table.scrollToBottom()

    def _process_image_batch(self):
        """批量处理图片"""
        self._stop_video_capture()

        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "./")
        if not folder:
            return

        self.source_path = folder
        valid_extensions = {'jpg', 'png', 'jpeg', 'bmp'}

        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if (os.path.isfile(filepath) and
                    filename.split('.')[-1].lower() in valid_extensions):
                self._process_single_image(filepath)
                QApplication.processEvents()  # 保持UI响应

    def _update_selection(self):
        """更新用户选择的检测目标显示"""
        selection = self.ui.comboBox.currentText()

        if selection == '全部':
            boxes = self.detection_boxes
            display_img = self.current_result
            self._show_detection_details(0)
        else:
            idx = int(selection.split('_')[-1])
            boxes = [self.detection_boxes[idx]]
            display_img = self.detector(self.source_path)[0][idx].plot()
            self._show_detection_details(idx)

        # 更新显示
        width, height = self._calculate_display_size(display_img)
        resized_img = cv2.resize(display_img, (width, height))
        qimage = tools.cvimg_to_qpiximg(resized_img)

        self.ui.label_show.clear()
        self.ui.label_show.setPixmap(qimage)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

    def _handle_video_input(self):
        """处理视频输入"""
        if self.camera_active:
            self._toggle_camera()

        video_path = self._get_video_path()
        if not video_path:
            return

        self._start_video_processing(video_path)
        self.ui.comboBox.setEnabled(False)

    def _get_video_path(self):
        """获取视频文件路径"""
        path, _ = QFileDialog.getOpenFileName(
            self, '选择视频', './', "视频文件 (*.avi *.mp4)")

        if path:
            self.source_path = path
            self.ui.VideolineEdit.setText(path)
            return path
        return None

    def _start_video_processing(self, video_path):
        """开始处理视频流"""
        self.video_capture = cv2.VideoCapture(video_path)
        self.frame_timer.start(1)

    def _stop_video_capture(self):
        """停止视频捕获"""
        if self.video_capture:
            self.video_capture.release()
            self.frame_timer.stop()
            self.camera_active = False
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.video_capture = None

    def _process_video_frame(self):
        """处理视频帧"""
        ret, frame = self.video_capture.read()
        if not ret:
            self._stop_video_capture()
            return

        # 执行目标检测
        start_time = time.time()
        results = self.detector(frame)[0]
        processing_time = time.time() - start_time

        # 解析结果
        self.detection_boxes = results.boxes.xyxy.int().tolist()
        self.detection_classes = results.boxes.cls.int().tolist()
        self.confidence_scores = [f'{conf * 100:.2f}%' for conf in results.boxes.conf.tolist()]

        # 更新显示
        self._update_detection_display(results, processing_time)
        self._update_object_selection()
        self._show_detection_details()
        self._display_results_table(self.source_path)

    def _toggle_camera(self):
        """切换摄像头状态"""
        self.camera_active = not self.camera_active

        if self.camera_active:
            self.ui.CaplineEdit.setText('摄像头开启')
            self._start_video_processing(0)
            self.ui.comboBox.setEnabled(False)
        else:
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.ui.label_show.clear()
            self._stop_video_capture()

    def _save_results(self):
        """保存检测结果"""
        if not self.video_capture and not self.source_path:
            QMessageBox.information(self, '提示', '没有可保存的内容，请先打开图片或视频！')
            return

        if self.camera_active:
            QMessageBox.information(self, '提示', '无法保存摄像头实时视频！')
            return

        if self.video_capture:
            self._save_video_result()
        else:
            self._save_image_result()

    def _save_video_result(self):
        """保存视频检测结果"""
        confirm = QMessageBox.question(
            self, '确认',
            '保存视频可能需要较长时间，确定继续吗？',
            QMessageBox.Yes | QMessageBox.No)

        if confirm == QMessageBox.No:
            return

        self._stop_video_capture()
        saver = VideoSaverThread(
            self.source_path, self.detector,
            self.ui.comboBox.currentText())
        saver.start()
        saver.update_ui_signal.connect(self._update_progress)

    # def _save_image_result(self):
    #     """保存图片检测结果"""
    #     if os.path.isfile(self.source_path):
    #         # 处理单张图片
    #         filename = os.path.basename(self.source_path)
    #         name, ext = filename.rsplit(".", 1)
    #         save_name = f"{name}_detect_result.{ext}"
    #         save_path = os.path.join(Config.save_path, save_name)

    #         cv2.imwrite(save_path, self.current_result)
    #         QMessageBox.information(
    #             self, '完成',
    #             f'图片已保存至: {save_path}')
    #     else:
    #         # 处理文件夹中的图片
    #         valid_exts = {'jpg', 'png', 'jpeg', 'bmp'}
    #         for filename in os.listdir(self.source_path):
    #             if filename.split('.')[-1].lower() in valid_exts:
    #                 filepath = os.path.join(self.source_path, filename)
    #                 name, ext = filename.rsplit(".", 1)
    #                 save_name = f"{name}_detect_result.{ext}"
    #                 save_path = os.path.join(Config.save_path, save_name)

    #                 results = self.detector(filepath)[0]
    #                 cv2.imwrite(save_path, results.plot())

    #         QMessageBox.information(
    #             self, '完成',
    #             f'所有图片已保存至: {Config.save_path}')
            # 假设加在单图保存逻辑之后
    def _save_image_result(self):
        """保存图片检测结果及CSV数据"""
        # 定义 CSV 文件路径和表头
        csv_path = os.path.join(Config.save_path, "detect_results.csv")
        headers = ["ID", "路径", "类别", "置信度", "坐标位置"]

        if os.path.isfile(self.source_path):
            # ----------------- 1. 处理单张图片 -----------------
            filename = os.path.basename(self.source_path)
            name, ext = filename.rsplit(".", 1)
            save_name = f"{name}_detect_result.{ext}"
            save_img_path = os.path.join(Config.save_path, save_name)

            # 保存图片
            cv2.imwrite(save_img_path, self.current_result)

            # 保存单图的 CSV 数据
            data_lines = []
            for idx, (box, cls_id, conf) in enumerate(zip(self.detection_boxes, self.detection_classes, self.confidence_scores)):
                data_lines.append([self.source_path, Config.CH_names[cls_id], conf, str(box)])
            
            if data_lines: # 如果检测到了裂缝才写入
                tools.insert_rows(csv_path, data_lines, headers)

            QMessageBox.information(
                self, '完成',
                f'图片及检测数据已保存至:\n{Config.save_path}')

        else:
            # ----------------- 2. 处理文件夹中的批量图片 -----------------
            valid_exts = {'jpg', 'png', 'jpeg', 'bmp'}
            batch_data_lines = [] # 用于收集批量处理产生的所有数据

            for filename in os.listdir(self.source_path):
                if filename.split('.')[-1].lower() in valid_exts:
                    filepath = os.path.join(self.source_path, filename)
                    name, ext = filename.rsplit(".", 1)
                    save_name = f"{name}_detect_result.{ext}"
                    save_img_path = os.path.join(Config.save_path, save_name)

                    # 执行检测并保存图片
                    results = self.detector(filepath)[0]
                    cv2.imwrite(save_img_path, results.plot())

                    # 提取这张图的检测数据并加入列表
                    boxes = results.boxes.xyxy.int().tolist()
                    classes = results.boxes.cls.int().tolist()
                    confs = [f'{score * 100:.2f}%' for score in results.boxes.conf.tolist()]
                    
                    for box, cls_id, conf in zip(boxes, classes, confs):
                        batch_data_lines.append([filepath, Config.CH_names[cls_id], conf, str(box)])

            # 将批量数据一次性写入 CSV
            if batch_data_lines:
                tools.insert_rows(csv_path, batch_data_lines, headers)

            QMessageBox.information(
                self, '完成',
                f'批量图片及数据已汇总保存至:\n{Config.save_path}')

    def _update_progress(self, current, total):
        """更新保存进度"""
        if current == 1:
            self.progress_dialog = ProgressBar(self)
            self.progress_dialog.show()

        if current >= total:
            self.progress_dialog.close()
            QMessageBox.information(
                self, '完成',
                f'视频已保存至: {Config.save_path}')
            return

        if not self.progress_dialog.isVisible():
            return

        percent = int(current / total * 100)
        self.progress_dialog.setValue(current, total, percent)
        QApplication.processEvents()


class VideoSaverThread(QThread):
    """视频保存线程"""
    update_ui_signal = pyqtSignal(int, int)

    def __init__(self, video_path, model, selection):
        super().__init__()
        self.video_path = video_path
        self.detector = model
        self.selection = selection
        self.active = True
        self.colors = tools.Colors()

    def run(self):
        """执行视频保存"""
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        filename = os.path.basename(self.video_path)
        name, _ = filename.split('.')
        save_path = os.path.join(
            Config.save_path,
            f"{name}_detect_result.avi")

        writer = cv2.VideoWriter(save_path, fourcc, fps, size)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while cap.isOpened() and self.active:
            current_frame += 1
            ret, frame = cap.read()

            if not ret:
                break

            # 执行检测
            results = self.detector(frame)[0]
            frame = results.plot()
            writer.write(frame)
            self.update_ui_signal.emit(current_frame, total_frames)

        # 释放资源
        cap.release()
        writer.release()

    def stop(self):
        """停止保存过程"""
        self.active = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())