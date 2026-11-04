# UAV Detection & TRACKING

Đây là module nhận diện và theo dõi (tracking) UAV trên không gian 2D (video phẳng) thuộc khuôn khổ bài tập lớn môn Xử lý ảnh. Module này hoạt động độc lập để kiểm thử độ chính xác của AI trước khi tích hợp vào luồng xử lý không gian 3D (Kinect Depth + Kalman Filter).

## 📌 Tính năng chính
- **Giao diện trực quan (GUI):** Tích hợp Tkinter cho phép chọn file video cần test trực tiếp từ thư mục `Testing`.
- **Nhận diện UAV thời gian thực:** Sử dụng trọng số YOLOv11s đã được fine-tune chuyên biệt cho drone.
- **Theo dõi đối tượng (Tracking):** Tích hợp thuật toán ByteTrack để gán ID và duy trì bounding box xuyên suốt các frame hình.

## 📂 Cấu trúc thư mục
```text
UAVDetection/
│
├── Testing/                 # Thư mục chứa các file video test (.mp4, .avi)
│   ├── UAV1.mp4
│   └── UAV2.mp4
│   └── ...
│   └── UAV9.mp4
│
├── .venv/                   # Môi trường ảo Python (Không push lên Git)
├── best.pt                  # File trọng số model YOLOv11s (Tải từ Hugging Face)
├── demo_uav.py              # Script chính khởi chạy giao diện và luồng tracking
├── requirements.txt         # Danh sách thư viện phụ thuộc
└── README.md                # Tài liệu hướng dẫn module
```

## 🛠 Cài đặt môi trường

**Bước 1: Khởi tạo môi trường ảo (Khuyến nghị)**
Mở Terminal tại thư mục `UAVDetection` và chạy:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Bước 2: Cài đặt thư viện**
```powershell
pip install -r requirements.txt
```
*(Thư viện chính: `ultralytics`, Tkinter có sẵn trong Python chuẩn).*

**Bước 3: Tải Model Trọng số**
- Đảm bảo file `best.pt` (hoặc tên tương ứng thiết lập trong code) đã được tải về và đặt tại thư mục gốc `UAVDetection`.
- Nguồn model: [sapoepsilon/yolov11s-drone-detector](https://huggingface.co/sapoepsilon/yolov11s-drone-detector)

## 🚀 Hướng dẫn sử dụng

Chạy lệnh sau trên terminal để mở giao diện kiểm thử:
```powershell
python demo_uav.py
```
- Hệ thống sẽ tự động quét thư mục `Testing` và hiển thị danh sách video.
- Chọn video cần test và bấm **Bắt đầu Tracking**.
- Kết quả video tracking kèm Bounding Box & ID sẽ được tự động lưu vào thư mục `runs/detect/track`.

## 🔄 Hướng phát triển và Tích hợp (Next Steps)
Trong giai đoạn tới, module này sẽ được kết nối với module `KinectTennisTracker` của nhóm trưởng theo lộ trình:
1. **Trích xuất tọa độ tâm:** Tùy chỉnh mã nguồn để trả về tọa độ `(x, y)` của bounding box thay vì chỉ vẽ lên video.
2. **Chuyển đổi ONNX:** Export model `.pt` sang `.onnx` (`yolo export model=best.pt format=onnx`) để chạy chung thư viện `onnxruntime`, tối ưu hóa hiệu suất luồng.
3. **Truyền dữ liệu 3D:** Đẩy tọa độ `(x, y)` sang module Depth để lấy giá trị `Z` (chiều sâu) và đưa vào bộ lọc Kalman.