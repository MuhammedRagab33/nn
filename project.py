import cv2
import pytesseract
import re
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

# ضبط مسار Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class FastNetWeightScannerApp(App):
    def build(self):
        # إعداد واجهة المستخدم
        self.layout = BoxLayout(orientation="vertical")

        # إضافة الكاميرا
        self.camera = Camera(play=True, resolution=(640, 480))
        self.camera.allow_stretch = True
        self.layout.add_widget(self.camera)

        # عرض الوزن المكتشف
        self.weight_label = Label(
            text="Detected Net Weight: None", size_hint=(1, 0.1), font_size=20
        )
        self.layout.add_widget(self.weight_label)

        # منطقة نص لحفظ الأوزان
        self.draft_area = TextInput(
            hint_text="Saved Weights...", multiline=True, readonly=True, size_hint=(1, 0.3)
        )
        self.layout.add_widget(self.draft_area)

        # زر حفظ الوزن
        self.save_button = Button(text="Save Net Weight", size_hint=(1, 0.1))
        self.save_button.bind(on_press=self.save_weight)
        self.layout.add_widget(self.save_button)

        # بدء المعالجة التلقائية
        self.detected_weight = None
        Clock.schedule_interval(self.process_frame, 1.0 / 2.0)  # معالجة كل نصف ثانية

        return self.layout

    def process_frame(self, dt):
        # التقاط صورة من الكاميرا
        texture = self.camera.texture
        frame = np.frombuffer(texture.pixels, np.uint8).reshape(
            texture.height, texture.width, 4
        )

        # تحويل الصورة إلى تدرج الرمادي
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # تحديد منطقة الوزن (ROI)
        height, width = gray.shape
        roi = gray[int(height * 0.7): int(height * 0.85), int(width * 0.6): int(width * 0.95)]

        # تحسين الصورة باستخدام Threshold
        _, roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)

        # استخدام Tesseract لاستخراج النصوص
        extracted_text = pytesseract.image_to_string(roi, lang="eng")
        print("Extracted Text:", extracted_text)

        # استخراج الوزن الصافي باستخدام Regex
        match = re.search(r"(?:Net\s*Weight|Peso\s*Líquido).*?([\d.]+)\s*Kg", extracted_text, re.IGNORECASE)
        if match:
            new_weight = match.group(1)
            if new_weight != self.detected_weight:  # تحديث فقط إذا كان الوزن جديدًا
                self.detected_weight = new_weight
                self.weight_label.text = f"Detected Net Weight: {self.detected_weight} Kg"
        else:
            self.weight_label.text = "Detected Net Weight: None"

    def save_weight(self, instance):
        # حفظ الوزن في المسودة
        if self.detected_weight:
            self.draft_area.text += f"Net Weight: {self.detected_weight} Kg\n"
            self.weight_label.text = "Detected Net Weight: None"
            self.detected_weight = None


# تشغيل التطبيق
if __name__ == "__main__":
    FastNetWeightScannerApp().run()
