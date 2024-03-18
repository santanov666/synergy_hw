import sys
import cv2
import torch
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QAction, QFileDialog

# Загрузка модели YOLOv5
def load_model(model_path):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print(f'[INFO] Using device {device}')

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)

    if cuda:
        model.to(device)

    # Установка режима оценки (evaluation mode)
    model.eval()

    # Автоматическое изменение размеров входных изображений
    model.conf = 0.25
    model.autoscale = True

    return model, device


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, model, device, camera=0):
        super().__init__()
        self._run_flag = True
        self.model = model
        self.device = device
        self.camera = camera

    def run(self):
        # Инициализация видеопотока с веб-камеры
        cap = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW)
        # Установка разрешения и частоты кадров
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)
        # Установка формата видео на MJPG
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Создание текста для каждого класса объектов
        class_texts = [''] * len(self.model.names)
        for i, name in enumerate(self.model.names):
            class_texts[i] = f'{name}: '

        while self._run_flag:
            # Считывание кадра из видеопотока
            ret, frame = cap.read()

            if ret:
                with torch.no_grad():
                    # Применение модели YOLOv5 к текущему кадру
                    results = self.model(frame)

                    # Отображение результата на кадре
                    phone_detected = False  # Флаг обнаружения телефона в кадре
                    for detection in results.xyxy[0]:
                        x1, y1, x2, y2, conf, cls = detection.tolist()
                        label = self.model.names[int(cls)]
                        class_text = class_texts[int(cls)]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label}: {conf:.2f}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        # Проверяем, является ли найденный объект телефоном
                        if label == 'cell phone':
                            phone_detected = True

                    if phone_detected:
                        # Если в кадре обнаружен телефон, выводим текст с надписью "Alarm!!!" в нижнем углу экрана
                        cv2.putText(frame, "Alarm!!!", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Отображение кадра в интерфейсе приложения
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                    self.change_pixmap_signal.emit(p)

                # Освобождение использованного кадра
                del frame

                # Если пользователь нажал клавишу 'q', выходим из цикла и освобождаем ресурсы
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Object Detection'
        self.left = 0
        self.top = 0
        self.width = 640
        self.height = 480
        self.model_path = ''

        # Создание меню с возможностью выбора модели
        open_model_action = QAction('&Open Model', self)
        open_model_action.triggered.connect(self.open_model_dialog)

        self.menuBar().addMenu('File').addAction(open_model_action)

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Создание QLabel для отображения видеопотока
        self.label = QLabel(self)
        self.label.setScaledContents(True)
        self.label.resize(self.width, self.height)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        # Создание виджета для размещения QLabel
        central_widget = QWidget(self)
        lay = QVBoxLayout(central_widget)
        lay.addWidget(self.label)
        self.setCentralWidget(central_widget)

        # Если путь к модели не выбран, выводим диалоговое окно для выбора пути к модели
        if not self.model_path:
            self.open_model_dialog()

        # Загрузка модели YOLOv5
        self.model, self.device = load_model(self.model_path)

        # Запуск потока видеопотока
        self.thread = VideoThread(self.model, self.device)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    # Обновление изображения на QLabel
    @pyqtSlot(QImage)
    def update_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    # Закрытие приложения
    def closeEvent(self, event):
        self.thread._run_flag = False
        self.thread.wait()

    # Диалоговое окно для выбора пути к модели
    def open_model_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        model_path, _ = QFileDialog.getOpenFileName(self,"Выберите модель", "","All Files (*);;Python Files (*.pt)", options=options)
        if model_path:
            self.model_path = model_path

            # Если приложение уже запущено, перезагрузка модели
            if hasattr(self, 'model'):
                self.model, self.device = load_model(self.model_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

