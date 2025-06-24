import customtkinter as ctk
from PIL import Image, ImageTk
import threading
from tkinter import messagebox, Toplevel
from ais_module import AISClassifier, recognize_text_from_image, X_train, X_test, y_train, y_test, vectorizer, class_names, ais, svd

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class ImagePreview(Toplevel):
    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.title("Предпросмотр изображения")
        self.attributes('-topmost', True)
        
        # Переменные для перетаскивания окна
        self._drag_data = {"x": 0, "y": 0, "item": None}
        
        # Загружаем и отображаем изображение в полном размере
        img = Image.open(image_path)
        img_tk = ImageTk.PhotoImage(img)
        
        # Создаем canvas с размерами изображения
        self.canvas = ctk.CTkCanvas(self, width=img.width, height=img.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk  # Сохраняем ссылку
        
        # Устанавливаем размер окна по размеру изображения
        self.geometry(f"{img.width}x{img.height}")
        
        # Добавляем обработчик закрытия окна
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Привязываем события мыши для перетаскивания
        self.bind("<Button-1>", self.start_drag)
        self.bind("<B1-Motion>", self.drag)
        self.bind("<ButtonRelease-1>", self.stop_drag)
        
    def start_drag(self, event):
        # Запоминаем начальную позицию
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
        
    def drag(self, event):
        # Вычисляем смещение
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        
        # Получаем текущую позицию окна
        x = self.winfo_x() + dx
        y = self.winfo_y() + dy
        
        # Перемещаем окно
        self.geometry(f"+{x}+{y}")
        
    def stop_drag(self, event):
        # Очищаем данные перетаскивания
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0
        
    def on_closing(self):
        self.master.preview_window = None
        self.destroy()

class AIS_GUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AIS Текстовая Классификация")
        self.geometry("1400x900")
        self.minsize(1100, 700)
        self.image_path = None
        self.ocr_text = ""
        self.preview_window = None
        self.preview_timer = None

        # Основной split-контейнер
        main_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#f4f6fa")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Левая половина — предпросмотр фото
        left_panel = ctk.CTkFrame(main_frame, width=600, corner_radius=18, fg_color="#e3eaf2")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
        ctk.CTkLabel(left_panel, text="Исходное изображение", font=("Arial", 18, "bold"), text_color="#1e293b").pack(pady=(20, 10))
        
        # Создаем фрейм для canvas с подсказкой
        canvas_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        canvas_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.image_canvas = ctk.CTkCanvas(canvas_frame, width=560, height=700, bg="white", 
                                        highlightthickness=2, highlightbackground="#b0b0b0")
        self.image_canvas.pack(fill="both", expand=True)
        
        # Добавляем подсказку
        hint_label = ctk.CTkLabel(canvas_frame, text="Наведите курсор для просмотра в полном размере", 
                                 font=("Arial", 12), text_color="#666666")
        hint_label.pack(pady=(5, 0))
        
        # Привязываем события мыши
        self.image_canvas.bind("<Enter>", self.show_full_preview)
        self.image_canvas.bind("<Leave>", self.hide_full_preview)

        # Правая половина — результаты
        right_panel = ctk.CTkFrame(main_frame, corner_radius=18, fg_color="#ffffff")
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(right_panel, text="Результаты распознавания", font=("Arial", 20, "bold"), text_color="#1e293b").pack(anchor="w", padx=20, pady=(20, 0))
        # Рамка для распознанного текста и результатов
        self.ocr_frame = ctk.CTkFrame(right_panel, fg_color="#f9fafb", border_width=3, border_color="#3b82f6", corner_radius=14)
        self.ocr_frame.pack(fill="both", padx=20, pady=(10, 18), expand=True)
        self.ocr_conf_label = ctk.CTkLabel(self.ocr_frame, text="", font=("Arial", 20, "bold"), text_color="#e11d48")
        self.ocr_conf_label.pack(anchor="w", padx=16, pady=(16, 0))
        self.ocr_class_label = ctk.CTkLabel(self.ocr_frame, text="", font=("Arial", 18, "bold"), text_color="#2563eb")
        self.ocr_class_label.pack(anchor="w", padx=16, pady=(0, 10))
        # Прокручиваемое текстовое поле для распознанного текста
        self.ocr_textbox = ctk.CTkTextbox(self.ocr_frame, font=("Arial", 18), wrap="word", width=1, height=1)
        self.ocr_textbox.pack(fill="both", padx=16, pady=(0, 16), expand=True)
        self.ocr_textbox.configure(state="normal")
        self.ocr_textbox.delete("1.0", "end")
        self.ocr_textbox.insert("end", "👋 Добро пожаловать в AIS Текстовая Классификация!\n\n"
            "1. Обучите модель\n"
            "2. Загрузите изображение\n"
            "3. Распознайте и классифицируйте текст\n\n"
            "Модель различает следующие категории:\n"
            + "\n".join(f"- {name}" for name in class_names)
            + "\n\nНажмите 'Обучить AIS' для начала работы.")
        self.ocr_textbox.configure(state="disabled")

        # Кнопки в две строки
        btn_panel1 = ctk.CTkFrame(right_panel, fg_color="#ffffff")
        btn_panel1.pack(fill="x", padx=20, pady=(0, 0))
        self._add_icon_button(btn_panel1, "Обучить AIS", "🧬", self.train_ais)
        self._add_icon_button(btn_panel1, "Загрузить изображение", "🖼️", self.load_image)
        self._add_icon_button(btn_panel1, "Распознать текст", "🔍", self.run_ocr)
        btn_panel2 = ctk.CTkFrame(right_panel, fg_color="#ffffff")
        btn_panel2.pack(fill="x", padx=20, pady=(0, 10))
        self._add_icon_button(btn_panel2, "Классифицировать текст", "📑", self.classify_recognized_text)
        self._add_icon_button(btn_panel2, "Сохранить модель", "💾", self.save_model)
        self._add_icon_button(btn_panel2, "Загрузить модель", "📂", self.load_model)

        # Прогресс-бар и статус-бар
        self.progress_var = ctk.DoubleVar()
        self.progress_bar = ctk.CTkProgressBar(right_panel, variable=self.progress_var, width=600, height=18)
        self.progress_bar.pack(fill="x", padx=20, pady=5)
        self.status_var = ctk.StringVar()
        self.status_bar = ctk.CTkLabel(right_panel, textvariable=self.status_var, anchor="w", font=("Arial", 12), fg_color="#e6e6e6", text_color="#333")
        self.status_bar.pack(fill="x", padx=20, pady=(0, 10))

    def _add_icon_button(self, parent, text, icon, command):
        btn = ctk.CTkButton(parent, text=f"{icon}  {text}", command=command, height=48, width=220,
                            font=("Arial", 15, "bold"), fg_color="#3b82f6", hover_color="#2563eb",
                            text_color="#fff", corner_radius=12)
        btn.pack(side="left", padx=8, pady=8, expand=True)

    def show_full_preview(self, event):
        # Проверяем, что событие произошло именно на canvas с изображением
        if event.widget != self.image_canvas:
            return
            
        if self.preview_timer:
            self.after_cancel(self.preview_timer)
            self.preview_timer = None
            
        if self.image_path and not self.preview_window:
            self.preview_window = ImagePreview(self, self.image_path)
            # Привязываем событие Leave только к canvas в окне предпросмотра
            self.preview_window.canvas.bind("<Leave>", self.hide_full_preview)

    def hide_full_preview(self, event=None):
        # Проверяем существование окна предпросмотра
        if not self.preview_window:
            return
            
        # Проверяем, что событие произошло именно на canvas в окне предпросмотра
        if event and hasattr(event, 'widget') and event.widget != self.preview_window.canvas:
            return
            
        if self.preview_window:
            # Добавляем небольшую задержку перед закрытием
            self.preview_timer = self.after(100, self._close_preview)

    def _close_preview(self):
        if self.preview_window:
            try:
                self.preview_window.destroy()
            except:
                pass  # Игнорируем ошибки при уничтожении окна
            finally:
                self.preview_window = None
        self.preview_timer = None

    def run_ocr(self):
        if not self.image_path:
            messagebox.showwarning("Внимание", "Сначала загрузите изображение!")
            return
        self.update_status("Распознавание текста...")
        self.progress_var.set(0)
        def ocr_process():
            text, confidence = recognize_text_from_image(self.image_path)
            self.ocr_text = text.strip()
            self.progress_var.set(100)
            self.ocr_conf_label.configure(text=f"Уверенность: {confidence:.2%}")
            self.ocr_textbox.configure(state="normal")
            self.ocr_textbox.delete("1.0", "end")
            
            # Правильная обработка абзацев
            # Разбиваем текст на строки и обрабатываем пустые строки
            lines = text.split('\n')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if line:  # Если строка не пустая
                    formatted_lines.append(line)
                else:  # Если строка пустая, добавляем двойной перенос
                    formatted_lines.append('')
                    formatted_lines.append('')
            
            # Собираем текст обратно, сохраняя форматирование
            formatted_text = '\n'.join(formatted_lines)
            
            self.ocr_textbox.insert("end", formatted_text)
            self.ocr_textbox.configure(state="disabled")
            self.ocr_class_label.configure(text="")
            self.update_status(f"Текст успешно распознан (уверенность: {confidence:.2%})")
        threading.Thread(target=ocr_process).start()

    def classify_recognized_text(self):
        try:
            text = self.ocr_text.strip()
            if not text:
                messagebox.showwarning("Предупреждение", "Сначала распознайте текст из изображения!")
                return
            self.update_status("Классификация текста...")
            self.progress_var.set(0)
            text = text.lower()
            text = ' '.join(text.split())
            text_vector = vectorizer.transform([text])
            text_vector = svd.transform(text_vector)
            predictions, confidences = ais.predict(text_vector)
            if predictions is None or len(predictions) == 0:
                self.ocr_class_label.configure(text="❗ Не удалось классифицировать текст.")
                return
            prediction = int(predictions[0])
            if prediction < 0 or prediction >= len(class_names):
                self.ocr_class_label.configure(text="❗ Неизвестный класс.")
                return
            label = class_names[prediction]
            confidence = float(confidences[0])
            self.ocr_class_label.configure(text=f"Класс: {label} ")
            self.update_status("Классификация завершена")
            self.progress_var.set(100)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при классификации: {str(e)}")
            self.update_status("Ошибка при классификации")
            self.progress_var.set(0)

    def load_image(self):
        path = ctk.filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            self.image_path = path
            img = Image.open(path)
            max_width = 560
            max_height = 700
            ratio = min(max_width/img.size[0], max_height/img.size[1])
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor=ctk.NW, image=img_tk)
            self.image_canvas.image = img_tk
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            self.update_status(f"Изображение загружено: {path}")

    def train_ais(self):
        def training():
            try:
                # Очищаем все метки
                self.ocr_conf_label.configure(text="")
                self.ocr_class_label.configure(text="")
                self.ocr_textbox.configure(state="normal")
                self.ocr_textbox.delete("1.0", "end")
                self.ocr_textbox.insert("end", "⏳ Обучение модели...")
                self.ocr_textbox.configure(state="disabled")
                self.progress_var.set(0)
                
                # Обучаем модель
                ais.fit(X_train, y_train, class_names=class_names)
                
                # Обновляем интерфейс после успешного обучения
                self.ocr_textbox.configure(state="normal")
                self.ocr_textbox.delete("1.0", "end")
                self.ocr_textbox.insert("end", "✅ Обучение завершено!\n\n"
                    "Модель успешно обучена на датасете '20 Newsgroups'.\n"
                    "Теперь вы можете загрузить изображение и классифицировать текст.")
                self.ocr_textbox.configure(state="disabled")
                self.update_status("Обучение завершено")
                self.progress_var.set(100)
            except Exception as e:
                # В случае ошибки показываем сообщение в текстовом поле
                self.ocr_textbox.configure(state="normal")
                self.ocr_textbox.delete("1.0", "end")
                self.ocr_textbox.insert("end", f"❗ Ошибка при обучении: {str(e)}")
                self.ocr_textbox.configure(state="disabled")
                self.update_status("Ошибка при обучении")
                self.progress_var.set(0)
        threading.Thread(target=training).start()

    def show_model_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Управление моделью")
        dialog.geometry("300x150")
        ctk.CTkButton(dialog, text="Сохранить модель", command=lambda: [self.save_model(), dialog.destroy()]).pack(pady=10)
        ctk.CTkButton(dialog, text="Загрузить модель", command=lambda: [self.load_model(), dialog.destroy()]).pack(pady=10)
        ctk.CTkButton(dialog, text="Отмена", command=dialog.destroy).pack(pady=10)

    def save_model(self):
        try:
            path = ctk.filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
            if path:
                ais.save(path)
                self.ocr_text_label.configure(text=f"✅ Модель сохранена в: {path}")
                messagebox.showinfo("Успех", "Модель успешно сохранена!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении модели: {str(e)}")

    def load_model(self):
        try:
            path = ctk.filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
            if path:
                ais.load(path)
                self.ocr_textbox.configure(state="normal")
                self.ocr_textbox.delete("1.0", "end")
                self.ocr_textbox.insert("end", f"✅ Модель загружена из: {path}")
                self.ocr_textbox.configure(state="disabled")
                messagebox.showinfo("Успех", "Модель успешно загружена!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке модели: {str(e)}")

    def update_status(self, message):
        self.status_var.set(message)
        self.update_idletasks()

if __name__ == "__main__":
    app = AIS_GUI()
    app.mainloop()
