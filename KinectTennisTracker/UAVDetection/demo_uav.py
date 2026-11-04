import os
import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO

# Tạo thư mục Testing nếu chưa tồn tại
if not os.path.exists("Testing"):
    os.makedirs("Testing")

def run_tracking():
    selection = listbox.curselection()
    if not selection:
        messagebox.showwarning("Cảnh báo", "Vui lòng chọn 1 video từ danh sách!")
        return
    
    # Lấy tên file và tạo đường dẫn tương đối
    video_name = listbox.get(selection[0])
    video_path = os.path.join("Testing", video_name)
    
    print(f"Đang khởi động YOLOv11s cho video: {video_name}...")
    
    # Tải mô hình và chạy tracking
    model = YOLO("best.pt")
    model.track(source=video_path, tracker="bytetrack.yaml", show=True, save=True)

# Khởi tạo cửa sổ chính
root = tk.Tk()
root.title("UAV Tracking - Demo GUI")
root.geometry("400x350")

# Frame chứa danh sách video
frame = tk.Frame(root)
frame.pack(pady=10, fill=tk.BOTH, expand=True)

tk.Label(frame, text="Danh sách video trong thư mục 'Testing':", font=("Arial", 10, "bold")).pack(anchor="w", padx=20)

# Listbox hiển thị danh sách
listbox = tk.Listbox(frame, width=50, height=12, selectbackground="#0078D7")
listbox.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)

# Tự động quét và hiển thị file video vào Listbox
for file in os.listdir("Testing"):
    if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
        listbox.insert(tk.END, file)

# Nút thực thi
run_btn = tk.Button(root, text="Bắt đầu Tracking", command=run_tracking, bg="#28A745", fg="white", font=("Arial", 12, "bold"))
run_btn.pack(pady=15)

root.mainloop()