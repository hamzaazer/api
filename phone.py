import flet as ft
import os
import requests

# IMPORTANT:
# - If API runs on your PC in same Wi-Fi: use PC IP like "http://192.168.1.10:8000"
# - If Android emulator: use "http://10.0.2.2:8000"
API_URL = "http://10.0.2.2:8000"

CLASS_NAMES = ['f', 'm', 'n', 'q', 's', 'v']


def main(page: ft.Page):
    page.title = "ECG (Android Client)"
    page.scroll = "auto"

    status = ft.Text()
    results_col = ft.Column(scroll="auto")
    picked_label = ft.Text("No image selected")

    selected_path = {"path": None}

    def on_pick_result(e: ft.FilePickerResultEvent):
        if e.files and len(e.files) > 0:
            # On Android (APK), FilePicker usually provides a usable path.
            selected_path["path"] = e.files[0].path
            picked_label.value = os.path.basename(selected_path["path"] or e.files[0].name)
            status.value = "📁 Image selected"
        else:
            status.value = "❌ No file selected"
        page.update()

    file_picker = ft.FilePicker(on_result=on_pick_result)
    page.overlay.append(file_picker)

    def pick_image(_):
        file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=["png", "jpg", "jpeg", "bmp"],
        )

    def run_predict(_):
        p = selected_path["path"]
        if not p or not os.path.exists(p):
            status.value = "❌ Select an image first"
            page.update()
            return

        try:
            with open(p, "rb") as f:
                files = {"file": (os.path.basename(p), f, "application/octet-stream")}
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
                        padding=10,
                        margin=6,
                        border=ft.border.all(1),
                        content=ft.Column([
                            ft.Text("BEST MODEL" if is_best else f"Model: {item['name']}", weight="bold"),
                            ft.Text(f"Prediction: {item['label']} (confidence: {item['confidence']:.2f})"),
                            ft.Text(" | ".join(f"{c}:{probs[c]*100:.1f}%" for c in CLASS_NAMES), size=12),
                        ]),
                    )
                )

            status.value = f"✅ Best model: {data['best_model']}"
        except Exception as ex:
            status.value = f"❌ API error: {ex}"

        page.update()

    page.add(
        ft.Text("🫀 ECG Android Client", size=22, weight="bold"),
        ft.ElevatedButton("Pick ECG Image", on_click=pick_image),
        picked_label,
        ft.ElevatedButton("Run Predict (API)", on_click=run_predict),
        ft.Divider(),
        status,
        ft.Text("Results:", weight="bold"),
        results_col,
    )


ft.app(target=main)