# Kinect StereoVisionProject — model 1517

> **🎥 Video Demo Hệ Thống:**
> 
> [![Xem video demo hệ thống Kinect StereoVisionProject](assets/demo_preview.png)](assets/Desktop 2026.11.04 - 21.09.23.01.mp4)
> 
> *(Hoặc bạn có thể xem trực tiếp video nhúng bên dưới)*
> 
> <p align="center">
>   <video width="100%" controls>
>     <source src="assets/Desktop 2026.11.04 - 21.09.23.01.mp4" type="video/mp4">
>     Trình duyệt của bạn không hỗ trợ thẻ video.
>   </video>
> </p>

---

**Bản đo vùng/căn chỉnh/vật cản mới:** chạy `depth_lab.py` hoặc chọn **Kinect 1517 - Measurement Live** trong PyCharm. Xem [INTEGRATION.md](INTEGRATION.md) để đo bằng chuột, hiệu chuẩn sàn, xuất CSV và ghép detector/OCR. `main.py --lab` cũng mở bản mới; `main.py` không tham số giữ giao diện cũ.

Ứng dụng Python hiển thị **hai video thời gian thực trong cùng một cửa sổ**:

- Trái: ảnh màu RGB (bộ nhớ BGR của OpenCV), 640 × 480.
- Phải: depth 640 × 480, thang màu cố định từ 0,8 đến 4,0 mét; điểm không đo được hoặc ngoài khoảng hiển thị có màu đen. Tâm dấu cộng cho biết khoảng cách tại điểm giữa ảnh depth.

Model **1517 là Kinect for Windows v1**, dùng **Microsoft Kinect for Windows SDK 1.8** và `Kinect10.dll`. Không dùng SDK 2.0 / PyKinect2. Chương trình gọi trực tiếp SDK bằng `ctypes`, tương thích Python 3.10–3.13 trên Windows mà không phụ thuộc wrapper Python 2 cũ.

Kinect cung cấp RGB và depth đo bằng cảm biến hồng ngoại structured light. Đây là ứng dụng RGB-D; không tính disparity từ cặp camera RGB trái/phải. Hai ảnh có góc nhìn khác nhau, chưa đăng ký tọa độ từng pixel. Không dùng cùng tọa độ RGB và depth để suy ra cùng một vật thể. Chương trình ghép khung dựa trên timestamp của thiết bị, lệch tối đa 40 ms; không tuyên bố đồng bộ phơi sáng chính xác.

## Chuẩn bị camera

1. Cài [Kinect for Windows SDK 1.8 từ Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=40278) theo hướng dẫn của bộ cài, rồi khởi động lại nếu được yêu cầu. SDK gồm driver và runtime. SDK cũ được Microsoft công bố hỗ trợ Windows 7/8/8.1; trên Windows 10/11 cần kiểm tra driver thực tế trên máy.
2. Cấp nguồn ngoài cho Kinect, cắm USB trực tiếp vào máy (ưu tiên cổng USB 2.0, tránh hub), và đợi Windows nhận thiết bị.
3. Đóng Kinect Studio và ứng dụng khác đang dùng cảm biến.
4. Chạy `python main.py --diagnose` để kiểm tra runtime và số cảm biến nhận được.

Không tự tải DLL lẻ từ website không rõ nguồn. Chế độ mô phỏng luôn có nhãn **SIMULATION** và chỉ chạy khi có `--demo`; chương trình không tự thay camera thật bằng dữ liệu giả.

## Chạy trong PyCharm

Project có thể dùng môi trường `.venv-kinect` riêng. Chọn interpreter:

`<thư mục project>\.venv-kinect\Scripts\python.exe`

Nếu chưa có môi trường, tạo Python virtualenv trong PyCharm (Python 3.10–3.13), sau đó cài:

```powershell
python -m pip install -r requirements.txt
```

Mở `main.py`, chọn **Run** để dùng Kinect thật. Hai cấu hình chạy có sẵn trong `.run` là **Kinect 1517 - Live** và **Kinect 1517 - Demo**. Hoặc dùng:

```powershell
.\.venv-kinect\Scripts\python.exe main.py
.\.venv-kinect\Scripts\python.exe main.py --demo
```

Trong project còn có `run_live.cmd` và `run_demo.cmd` để mở bằng cách nhấp đúp.

## Điều khiển

- **Q / Esc** hoặc đóng cửa sổ: thoát và giải phóng camera.
- **G**: chuyển depth giữa màu và xám.
- **S**: lưu một cặp ảnh vào `captures/<thời gian>/`. Có `color.png`, `depth_color.png`, **`depth_mm.png` 16-bit** và `metadata.json` chứa timestamp, đơn vị, chế độ nguồn dữ liệu.

Đọc ảnh depth gốc bằng `cv2.imread(path, cv2.IMREAD_UNCHANGED)`; giá trị 2345 nghĩa là 2,345 mét. Giá trị 0 là không hợp lệ. Ảnh depth màu chỉ dùng quan sát, không dùng để đo khoảng cách.

Ví dụ tùy chọn:

```powershell
python main.py --near-mode --min-depth 400 --max-depth 3000
python main.py --timeout 15
python main.py --demo --headless --frames 60 --preview demo_preview.png
python -m unittest -v test_vision
```

`--near-mode` bật chế độ đo gần của Kinect for Windows v1; chỉ đổi thang màu bằng `--min-depth` không tự bật near mode. Khi không nhận đủ hai luồng trong 10 giây, chương trình báo lỗi và dừng; dùng `--timeout` để đổi thời gian này.

## Cấu trúc

- `main.py`: chạy ứng dụng, tham số, lưu ảnh và xử lý lỗi.
- `kinect_v1.py`: binding SDK 1.8, đọc texture theo stride và giải mã depth (`packed >> 3`).
- `vision.py`: ghép timestamp, xử lý depth, giao diện hai video và nguồn mô phỏng.
- `test_vision.py`: kiểm tra bộ nhớ SDK, giải mã depth, ghép khung và ảnh PNG depth 16-bit.

## Xử lý lỗi

- **Missing Kinect10.dll**: cài SDK 1.8; SDK 2.0 phục vụ đời Kinect khác.
- **Sensors: 0**: kiểm tra nguồn, dây USB và thiết bị Kinect trong Device Manager.
- **NuiInitialize / Open stream failed**: kiểm tra driver, đóng ứng dụng đang chiếm camera; giữ mã HRESULT khi báo lỗi.
- **No matching RGB/depth pair**: cảm biến không cấp đủ hai luồng hoặc đã mất kết nối. Kiểm tra dây, nguồn, cổng USB và driver.
- **Interpreter không chạy**: đổi interpreter PyCharm sang `.venv-kinect`; môi trường `.venv` cũ trong project đang trỏ tới Python Microsoft Store không chạy được ở lần kiểm tra này.

Tài liệu API: [Microsoft Kinect SDK 1.8](https://learn.microsoft.com/en-us/previous-versions/windows/kinect-1.8/hh855326(v=ieb.10)), [header Microsoft NuiImageCamera.h được Autodesk công bố](https://help.autodesk.com/cloudhelp/2018/ENU/MotionBuilder-SDK/cpp_ref/_nui_image_camera_8h_source.html).
