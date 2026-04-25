import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject, QRectF, QTimer, QUrl
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
    QCheckBox
)

MODEL_PATH = "model.pt"
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.3
CONF_THRESHOLD_LOCK = threading.Lock()
SOUND = "alert.wav"

GOOD_COLOR = QColor("#22c55e")
BAD_COLOR = QColor("#ef4444")
UNKNOWN_COLOR = QColor("#9ca3af")
BG_COLOR = QColor("#111827")
PLOT_BG_COLOR = QColor("#1f2937")
GRID_COLOR = QColor("#334155")
TEXT_COLOR = QColor("#e5e7eb")


def qimage_from_bgr(frame_bgr: 'cv2.Mat') -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, channel = rgb.shape
    bytes_per_line = channel * width
    return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()


@dataclass
class DetectionResult:
    frame: 'cv2.Mat'
    posture: str


class CameraPostureWorker(QObject):
    frame_ready = pyqtSignal(QImage)
    posture_ready = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path: str, camera_index: int = 0, conf_threshold: float = 0.45) -> None:
        super().__init__()
        self.model_path = model_path
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self._running = False
        self._conf_lock = threading.Lock()
        self._model = None
        self._cap = None

    @pyqtSlot()
    def run(self) -> None:
        if YOLO is None:
            self.error_occurred.emit(
                "Не удалось импортировать библиотеку 'ultralytics'.\n"
                "Установите её командой 'pip install ultralytics' и перезапустите программу."
            )
            self.status_changed.emit("Камера/модель: ошибка инициализации")
            return

        if not os.path.exists(self.model_path):
            self.error_occurred.emit(f"Файл модели не найден:\n{self.model_path}")
            self.status_changed.emit("Камера/модель: модель не найдена")
            return

        try:
            self.status_changed.emit("Загрузка модели...")
            self._model = YOLO(self.model_path)
        except Exception as e:
            self.error_occurred.emit(f"Ошибка загрузки модели:\n{e}")
            self.status_changed.emit("Камера/модель: ошибка модели")
            return

        try:
            self.status_changed.emit("Открытие камеры...")
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                raise RuntimeError(f"Не удалось открыть камеру (index={self.camera_index})")

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

            self._running = True
            self.status_changed.emit(f"Камера: активна | CONF={self.get_conf_threshold():.2f}")

            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    self.posture_ready.emit("unknown")
                    self.status_changed.emit("Камера: кадр недоступен")
                    QThread.msleep(80)
                    continue

                result = self._predict_posture(frame)

                self.posture_ready.emit(result.posture)
                qimg = qimage_from_bgr(result.frame)
                self.frame_ready.emit(qimg)

                QThread.msleep(10)

        except Exception as e:
            self.error_occurred.emit(f"Ошибка в потоке камеры/модели:\n{e}")
            self.status_changed.emit("Камера/модель: ошибка")
        finally:
            self._cleanup()

    def get_conf_threshold(self) -> float:
        with self._conf_lock:
            return self.conf_threshold

    @pyqtSlot(float)
    def set_conf_threshold(self, value: float) -> None:
        global CONF_THRESHOLD
        new_value = max(0.01, min(0.99, float(value)))
        with self._conf_lock:
            self.conf_threshold = new_value
        with CONF_THRESHOLD_LOCK:
            CONF_THRESHOLD = new_value
        self.status_changed.emit(f"CONF_THRESHOLD обновлён: {new_value:.2f}")

    def stop(self) -> None:
        self._running = False

    def _cleanup(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        self.status_changed.emit("Камера: остановлена")

    def _predict_posture(self, frame: 'cv2.Mat') -> DetectionResult:
        posture = "unknown"
        current_conf = self.get_conf_threshold()
        try:
            results = self._model.predict(source=frame, conf=current_conf, verbose=False)
            if not results or getattr(results[0], "boxes", None) is None or len(results[0].boxes) == 0:
                return DetectionResult(frame=frame.copy(), posture="unknown")

            result = results[0]
            boxes = result.boxes  
            names = getattr(result, "names", None) or getattr(self._model, "names", {})

            best_idx = max(range(len(boxes)), key=lambda i: float(boxes.conf[i]))

            xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            cls_id = int(boxes.cls[best_idx])
            cls_name: str
            if isinstance(names, dict):
                cls_name = names.get(cls_id, str(cls_id))
            else:
                cls_name = str(cls_id)

            if cls_name.lower() == "good_posture" or cls_id == 1:
                color = (0, 255, 0)
                posture = "good"
            elif cls_name.lower() == "bad_posture" or cls_id == 0:
                color = (0, 0, 255)
                posture = "bad"
            else:
                color = (255, 255, 0)
                posture = "unknown"

            frame_drawn = frame.copy()
            cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame_drawn,
                f"{posture.upper()} | CONF={current_conf:.2f}",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )

            return DetectionResult(frame=frame_drawn, posture=posture)
        except Exception:
            return DetectionResult(frame=frame.copy(), posture="unknown")


class PostureTimelineWidget(QWidget):
    def __init__(self, history_seconds: int = 120, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.history_seconds = history_seconds
        self._samples: deque[tuple[float, str]] = deque()
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def add_sample(self, timestamp: float, posture: str) -> None:
        posture = posture if posture in {"good", "bad", "unknown"} else "unknown"
        self._samples.append((timestamp, posture))
        self._prune(timestamp)
        self.update()

    def posture_percentages(self) -> dict[str, float]:
        durations = self._durations()
        total = sum(durations.values())
        if total <= 0:
            return {"good": 0.0, "bad": 0.0, "unknown": 0.0}
        return {key: (value / total) * 100.0 for key, value in durations.items()}

    def _prune(self, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        cutoff = now - self.history_seconds
        while len(self._samples) > 1 and self._samples[1][0] < cutoff:
            self._samples.popleft()

    def _durations(self) -> dict[str, float]:
        now = time.monotonic()
        self._prune(now)
        durations = {"good": 0.0, "bad": 0.0, "unknown": 0.0}
        if not self._samples:
            return durations

        points = list(self._samples)
        window_start = now - self.history_seconds

        if points[0][0] > window_start:
            points.insert(0, (window_start, points[0][1]))

        for idx, (start_ts, posture) in enumerate(points):
            end_ts = points[idx + 1][0] if idx + 1 < len(points) else now
            seg_start = max(start_ts, window_start)
            seg_end = min(end_ts, now)
            if seg_end > seg_start:
                durations[posture] = durations.get(posture, 0.0) + (seg_end - seg_start)
        return durations

    def _y_for_posture(self, plot_rect: QRectF, posture: str) -> float:
        if posture == "good":
            return plot_rect.top() + plot_rect.height() * 0.20
        if posture == "bad":
            return plot_rect.top() + plot_rect.height() * 0.80
        return plot_rect.center().y()

    def _color_for_posture(self, posture: str) -> QColor:
        if posture == "good":
            return GOOD_COLOR
        if posture == "bad":
            return BAD_COLOR
        return UNKNOWN_COLOR

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BG_COLOR)

        outer = self.rect().adjusted(10, 10, -10, -10)
        painter.setPen(QPen(QColor("#475569"), 1))
        painter.setBrush(PLOT_BG_COLOR)
        painter.drawRoundedRect(outer, 12, 12)

        left_pad = 74
        right_pad = 20
        top_pad = 24
        bottom_pad = 40
        plot_rect = QRectF(
            outer.left() + left_pad,
            outer.top() + top_pad,
            max(10, outer.width() - left_pad - right_pad),
            max(10, outer.height() - top_pad - bottom_pad),
        )

        now = time.monotonic()
        self._prune(now)

        painter.setPen(QPen(TEXT_COLOR, 1))
        title_font = painter.font()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(outer.adjusted(16, 6, -16, -6), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, f"История осанки за последние {self.history_seconds} сек")

        grid_pen = QPen(GRID_COLOR, 1)
        grid_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = plot_rect.top() + plot_rect.height() * frac
            painter.drawLine(int(plot_rect.left()), int(y), int(plot_rect.right()), int(y))

        for i in range(5):
            frac = i / 4
            x = plot_rect.left() + plot_rect.width() * frac
            painter.drawLine(int(x), int(plot_rect.top()), int(x), int(plot_rect.bottom()))

        painter.setPen(QPen(TEXT_COLOR, 1))
        painter.drawText(QRectF(outer.left() + 12, plot_rect.top() - 12, 55, 24), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "Хорошая")
        painter.drawText(QRectF(outer.left() + 12, plot_rect.center().y() - 12, 55, 24), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "Неизм.")
        painter.drawText(QRectF(outer.left() + 12, plot_rect.bottom() - 12, 55, 24), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "Плохая")

        small_font = painter.font()
        small_font.setPointSize(9)
        small_font.setBold(False)
        painter.setFont(small_font)
        painter.drawText(QRectF(plot_rect.left(), plot_rect.bottom() + 8, 70, 24), Qt.AlignmentFlag.AlignLeft, f"-{self.history_seconds} c")
        painter.drawText(QRectF(plot_rect.right() - 45, plot_rect.bottom() + 8, 45, 24), Qt.AlignmentFlag.AlignRight, "сейчас")

        if not self._samples:
            painter.setPen(QPen(QColor("#94a3b8"), 1))
            painter.drawText(plot_rect, Qt.AlignmentFlag.AlignCenter, "Недостаточно данных для построения графика")
            painter.end()
            return

        points = list(self._samples)
        window_start = now - self.history_seconds
        if points[0][0] > window_start:
            points.insert(0, (window_start, points[0][1]))

        def x_for_time(ts: float) -> float:
            return plot_rect.left() + ((ts - window_start) / self.history_seconds) * plot_rect.width()

        for idx, (start_ts, posture) in enumerate(points):
            end_ts = points[idx + 1][0] if idx + 1 < len(points) else now
            seg_start = max(start_ts, window_start)
            seg_end = min(end_ts, now)
            if seg_end <= seg_start:
                continue

            x1 = x_for_time(seg_start)
            x2 = x_for_time(seg_end)
            color = self._color_for_posture(posture)
            bg = QColor(color)
            bg.setAlpha(55)
            painter.fillRect(QRectF(x1, plot_rect.top(), max(1.0, x2 - x1), plot_rect.height()), bg)

        line_pen = QPen(QColor("#ffffff"), 2)
        painter.setPen(line_pen)
        prev_end_x = None
        prev_y = None
        for idx, (start_ts, posture) in enumerate(points):
            end_ts = points[idx + 1][0] if idx + 1 < len(points) else now
            seg_start = max(start_ts, window_start)
            seg_end = min(end_ts, now)
            if seg_end <= seg_start:
                continue

            x1 = x_for_time(seg_start)
            x2 = x_for_time(seg_end)
            y = self._y_for_posture(plot_rect, posture)
            segment_pen = QPen(self._color_for_posture(posture), 4)
            segment_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(segment_pen)
            painter.drawLine(int(x1), int(y), int(x2), int(y))

            if prev_end_x is not None and prev_y is not None and abs(x1 - prev_end_x) < 3:
                connector_pen = QPen(QColor("#cbd5e1"), 2)
                painter.setPen(connector_pen)
                painter.drawLine(int(prev_end_x), int(prev_y), int(x1), int(y))

            prev_end_x = x2
            prev_y = y

        painter.end()


class PostureWindow(QMainWindow):
    conf_changed = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Мониторинг осанки")
        self.resize(1200, 860)

        self._last_history_sample_ts = 0.0
        self._last_history_posture = "unknown"
        self._history_sample_interval = 0.25
        
        self._sound_alert_enabled = True
        self._bad_posture_active = False    

        self.alert_sound = QSoundEffect(self)                           
        self.alert_sound.setSource(QUrl.fromLocalFile(SOUND))     
        self.alert_sound.setVolume(0.7)                                 
        self.alert_sound.setLoopCount(-2)                               

        self.bad_posture_sound_timer = QTimer(self)                                 
        self.bad_posture_sound_timer.setSingleShot(True)                            
        self.bad_posture_sound_timer.setInterval(3000)                              
        self.bad_posture_sound_timer.timeout.connect(self.start_bad_posture_alert)  

        central = QWidget()
        self.setCentralWidget(central)
        self.setStyleSheet("background-color: #111827;")

        self.video_label = QLabel("Ожидание видеопотока...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet(
            "background-color: #000000; color: #D1D5DB; border: 1px solid #374151; border-radius: 10px;"
        )

        self.status_label = QLabel("Осанка: неизвестно")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(13)
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet(
            "color: #FFFFFF; background-color: #374151; padding: 8px; border-radius: 8px;"
        )

        self.conf_title_label = QLabel("Порог CONF")
        self.conf_title_label.setStyleSheet("color: #E5E7EB; font-weight: 600;")

        self.conf_value_label = QLabel(f"{CONF_THRESHOLD:.2f}")
        self.conf_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.conf_value_label.setMinimumWidth(56)
        self.conf_value_label.setStyleSheet("color: #93C5FD; font-weight: 700;")

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setSingleStep(1)
        self.conf_slider.setPageStep(5)
        self.conf_slider.setValue(int(CONF_THRESHOLD * 100))
        self.conf_slider.setTracking(True)
        self.conf_slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 8px; background: #374151; border-radius: 4px; }"
            "QSlider::handle:horizontal { background: #60A5FA; width: 18px; margin: -6px 0; border-radius: 9px; }"
            "QSlider::sub-page:horizontal { background: #2563EB; border-radius: 4px; }"
        )

        self.sound_checkbox = QCheckBox("Включить звуковые оповещения")
        self.sound_checkbox.setChecked(True)
        self.sound_checkbox.setStyleSheet(" color : #E5E7EB;")
        self.sound_checkbox.toggled.connect(self.on_sound_toggled)

        controls_frame = QFrame()
        controls_frame.setStyleSheet("background-color: #1F2937; border-radius: 10px;")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(14, 12, 14, 12)
        controls_layout.setSpacing(12)
        controls_layout.addWidget(self.conf_title_label)
        controls_layout.addWidget(self.conf_slider, stretch=1)
        controls_layout.addWidget(self.conf_value_label)

        self.dashboard_title = QLabel("Дэшборд")
        dash_font = QFont()
        dash_font.setPointSize(11)
        dash_font.setBold(True)
        self.dashboard_title.setFont(dash_font)
        self.dashboard_title.setStyleSheet("color: #F9FAFB;")

        self.timeline_widget = PostureTimelineWidget(history_seconds=120)

        self.summary_label = QLabel("Статистика за окно истории: хорошая 0.0% | плохая 0.0% | неизвестно 0.0%")
        self.summary_label.setStyleSheet("color: #CBD5E1; background-color: #1F2937; padding: 10px; border-radius: 10px;")

        legend_label = QLabel("Зелёный — хорошая, красный — плохая, серый — неизвестно")
        legend_label.setStyleSheet("color: #94A3B8;")

        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(self.video_label, stretch=5)
        layout.addWidget(self.status_label)
        layout.addWidget(controls_frame)
        layout.addWidget(self.sound_checkbox)
        layout.addWidget(self.dashboard_title)
        layout.addWidget(self.timeline_widget, stretch=3)
        layout.addWidget(legend_label)
        layout.addWidget(self.summary_label)

        self.camera_thread = QThread(self)
        self.camera_worker = CameraPostureWorker(
            model_path=MODEL_PATH,
            camera_index=CAMERA_INDEX,
            conf_threshold=CONF_THRESHOLD,
        )
        self.camera_worker.moveToThread(self.camera_thread)

        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.frame_ready.connect(self.update_frame)
        self.camera_worker.posture_ready.connect(self.update_posture)
        self.camera_worker.status_changed.connect(self.on_status_changed)
        self.camera_worker.error_occurred.connect(self.on_error)
        self.conf_changed.connect(self.camera_worker.set_conf_threshold)
        self.conf_slider.valueChanged.connect(self.on_conf_slider_changed)

        self.camera_thread.start()

    def start_bad_posture_alert(self) -> None:
        if self._sound_alert_enabled and self._bad_posture_active:
            if not self.alert_sound.isPlaying():
                self.alert_sound.play()

    @pyqtSlot(QImage)
    def update_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    @pyqtSlot(int)
    def on_conf_slider_changed(self, value: int) -> None:
        global CONF_THRESHOLD
        conf_value = value / 100.0
        self.conf_value_label.setText(f"{conf_value:.2f}")
        with CONF_THRESHOLD_LOCK:
            CONF_THRESHOLD = conf_value
        self.camera_worker.set_conf_threshold(conf_value)

    @pyqtSlot(bool)
    def on_sound_toggled(self, checked: bool) -> None:
        self._sound_alert_enabled = checked

        if not checked:
            self.bad_posture_sound_timer.stop()
            if self.alert_sound.isPlaying():
                self.alert_sound.stop()


    @pyqtSlot(str)
    def update_posture(self, posture: str) -> None:
        if posture == "good":
            self.status_label.setText("Осанка: правильная")
            self.status_label.setStyleSheet(
                "color: #22C55E; background-color: #1F2937; padding: 8px; border-radius: 8px;"
            )

            self._bad_posture_active = False
            self.bad_posture_sound_timer.stop()
            if self.alert_sound.isPlaying():
                self.alert_sound.stop()
        elif posture == "bad":
            self.status_label.setText("Осанка: неправильная")
            self.status_label.setStyleSheet(
                "color: #EF4444; background-color: #1F2937; padding: 8px; border-radius: 8px;"
            )

            self._bad_posture_active = True

            if (
                    self._sound_alert_enabled
                    and not self.alert_sound.isPlaying()
                    and not self.bad_posture_sound_timer.isActive()
            ):
                self.bad_posture_sound_timer.start()
        else:
            self.status_label.setText("Осанка: не определена")
            self.status_label.setStyleSheet(
                "color: #CBD5E1; background-color: #1F2937; padding: 8px; border-radius: 8px;"
            )

            self._bad_posture_active = False
            self.bad_posture_sound_timer.stop()
            if self.alert_sound.isPlaying():
                self.alert_sound.stop()

        now = time.monotonic()
        if (
            posture != self._last_history_posture
            or (now - self._last_history_sample_ts) >= self._history_sample_interval
        ):
            self.timeline_widget.add_sample(now, posture)
            self._last_history_sample_ts = now
            self._last_history_posture = posture
            self._refresh_summary()

    def _refresh_summary(self) -> None:
        stats = self.timeline_widget.posture_percentages()
        self.summary_label.setText(
            "Статистика за окно истории: "
            f"хорошая {stats['good']:.1f}% | "
            f"плохая {stats['bad']:.1f}% | "
            f"неизвестно {stats['unknown']:.1f}%"
        )

    @pyqtSlot(str)
    def on_status_changed(self, text: str) -> None:
        self.statusBar().showMessage(text)

    @pyqtSlot(str)
    def on_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)

    def closeEvent(self, event) -> None:
        try:
            self.camera_worker.stop()
        except Exception:
            pass
        self.camera_thread.quit()
        self.camera_thread.wait(2000)
        super().closeEvent(event)



def main() -> None:
    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    window = PostureWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()