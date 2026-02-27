import flet as ft
import os
import requests
from tkinter import Tk, filedialog

API_URL = "http://127.0.0.1:8000"

CLASS_NAMES = ['f', 'm', 'n', 'q', 's', 'v']


def main(page: ft.Page):
    page.title = "ECG Multi-Model Comparison (API)"
    page.window_width = 520
    page.window_height = 720

    image_path = None

    status = ft.Text()
    results_col = ft.Column(scroll="auto", expand=True)
    image_label = ft.Text("No image selected")

    def pick_image(e):
        nonlocal image_path
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select ECG image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        root.destroy()

        if file_path:
            image_path = file_path
            image_label.value = os.path.basename(file_path)
            status.value = "📁 ECG image selected"
            page.update()

    

    def run_comparison(e):
        if not image_path or not os.path.exists(image_path):
            status.value = "❌ Upload an ECG image"
            page.update()
            return

        try:
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f, "application/octet-stream")}
                r = requests.post(f"{API_URL}/predict", files=files, timeout=60)
            r.raise_for_status()
            data = r.json()

            results_col.controls.clear()

            best_name = data["best_model"]
            for item in data["results"]:
                is_best = (item["name"] == best_name)
                probs = item["probs"]

                results_col.controls.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Text("BEST MODEL" if is_best else f"Model: {item['name']}", weight="bold"),
                            ft.Text(f"Prediction: {item['label']} (confidence: {item['confidence']:.2f})"),
                            ft.Text(
                                " | ".join(f"{c}:{probs[c]*100:.1f}%" for c in CLASS_NAMES),
                                size=12
                            ),
                        ]),
                        border=ft.border.all(1),
                        padding=10,
                        margin=6
                    )
                )

            status.value = f"Best model: {data['best_model']}"
        except Exception as ex:
            status.value = f"❌ Prediction error: {ex}"

        page.update()

    page.add(
        ft.Text("🫀 ECG Multi-Model Comparison (API)", size=22, weight="bold"),
        
        ft.ElevatedButton("Upload ECG Image", on_click=pick_image),
        image_label,
        ft.ElevatedButton("Run Comparison (API)", on_click=run_comparison),
        ft.Divider(),
        status,
        ft.Text("Results:", weight="bold"),
        results_col
    )


ft.app(target=main)