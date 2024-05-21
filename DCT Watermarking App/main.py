import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from utils import embed_watermark, extract_watermark

class WatermarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DCT Watermarking App")
        self.root.configure(bg='#2c3e50')

        self.title_label = tk.Label(root, text="DCT Watermarking App", font=("Helvetica", 18, "bold"), bg='#2c3e50', fg='white')
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.img_frame = tk.Frame(root, width=400, height=400, bg='#34495e', relief='sunken', bd=2)
        self.img_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        self.img_frame.grid_propagate(False)

        self.plus_label = tk.Label(self.img_frame, text="+", font=("Helvetica", 72), bg='#34495e', fg='white')
        self.plus_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.img_label = tk.Label(self.img_frame)
        self.img_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        button_style = {
            "font": ("Helvetica", 12),
            "bg": "#3498db",
            "fg": "white",
            "activebackground": "#2980b9",
            "activeforeground": "white",
            "relief": "raised",
            "bd": 2,
            "width": 20
        }

        self.load_button = tk.Button(root, text="LOAD IMAGE", command=self.load_image, **button_style)
        self.load_button.grid(row=2, column=0, padx=5, pady=10)

        self.reset_button = tk.Button(root, text="RESET IMAGE", command=self.reset_image, **button_style)
        self.reset_button.grid(row=2, column=1, padx=5, pady=10)

        self.extract_button = tk.Button(root, text="EXTRACT WATERMARK", command=self.extract_watermark, **button_style)
        self.extract_button.grid(row=3, column=0, padx=5, pady=10)

        self.save_image_button = tk.Button(root, text="SAVE IMAGE", command=self.save_image, **button_style)
        self.save_image_button.grid(row=3, column=1, padx=5, pady=10)

        self.embed_button = tk.Button(root, text="EMBED WATERMARK", command=self.embed_watermark, **button_style)
        self.embed_button.grid(row=4, column=0, padx=5, pady=10)

        self.save_watermark_button = tk.Button(root, text="SAVE WATERMARK", command=self.save_watermark, **button_style)
        self.save_watermark_button.grid(row=4, column=1, padx=5, pady=10)

        self.original_image = None
        self.image = None
        self.extracted_watermark = None
        self.watermark = np.random.rand(32, 32) * 100

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.image = self.original_image.copy()
            self.display_image(self.image)

    def embed_watermark(self):
        if self.image is None:
            messagebox.showerror("Error", "Load an image first")
            return

        watermarked_image = embed_watermark(self.image, self.watermark)
        self.image = watermarked_image
        self.display_image(watermarked_image)
        messagebox.showinfo("Success", "Watermark embedded")

    def extract_watermark(self):
        if self.image is None:
            messagebox.showerror("Error", "Load an image first")
            return

        self.extracted_watermark = extract_watermark(self.image, self.watermark.shape)

        extracted_image = (self.extracted_watermark * 255).astype(np.uint8)
        extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_GRAY2RGB)

        self.display_image(extracted_image)

    def reset_image(self):
        if self.original_image is None:
            self.plus_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            self.img_label.config(image='')
            self.img_label.image = None
            messagebox.showinfo("Success", "Image reset")
        else:
            self.image = self.original_image.copy()
            self.display_image(self.image)
            messagebox.showinfo("Success", "Image reset")

    def save_image(self):
        if self.image is None:
            messagebox.showerror("Error", "No image to save")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, self.image)
            messagebox.showinfo("Success", "Image saved")

    def save_watermark(self):
        if self.extracted_watermark is None:
            messagebox.showerror("Error", "No watermark to save")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if file_path:
            watermark_image = (self.extracted_watermark * 255).astype(np.uint8)
            cv2.imwrite(file_path, watermark_image)
            messagebox.showinfo("Success", "Watermark saved")

    def display_image(self, image):
        self.plus_label.place_forget()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil.thumbnail((400, 400))
        image_tk = ImageTk.PhotoImage(image_pil)

        self.img_label.config(image=image_tk)
        self.img_label.image = image_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = WatermarkApp(root)
    root.mainloop()
