# Kinect Tennis Tracker

Project Windows tại `E:\KinectTennisTracker`. Kinect for Windows v1 model 1517 thu RGB và depth. YOLOv8n nhận diện bóng tennis trong ảnh. Tracker Kalman dùng các điểm depth đã căn theo RGB để ước lượng XYZ, vận tốc, hướng và vị trí dự kiến từ 0,1 đến 0,5 giây.

## Chạy trực tiếp

Kết nối Kinect với nguồn và USB. Mở `Start-Tennis.cmd` trong thư mục project. Nhấn **Q** hoặc **Esc** để dừng.

- Khung xanh quanh bóng cho biết có phép đo XYZ. Khung vàng và chữ `no depth` cho biết model thấy bóng trong ảnh nhưng vùng bóng thiếu dữ liệu depth đáng tin.
- Bên phải là hình chiếu XZ, tọa độ XYZ theo mét và vector hướng nếu vận tốc đủ lớn. Chấm lớn là vị trí hiện tại. Các chấm nhỏ là dự báo, cách nhau 0,1 giây. Vòng rỗng lớn đánh dấu track đang ngoại suy khi mất đo.
- Lần thử đầu nên dùng một bóng, giữ yên ở khoảng 1–1,5 m rồi dịch ngang từ từ. Để bóng tách rõ khỏi người cầm và khỏi tường phía sau. Cảnh rộng có thể có nhiều pixel depth trống mà vùng bóng vẫn đủ điểm.

```powershell
Set-Location E:\KinectTennisTracker
.\Start-Tennis.cmd
```

## Ghi kết quả hoặc thử riêng model

```powershell
# Phiên 15 giây, tạo dashboard 3D phát lại, CSV và thống kê JSON.
.\Start-Tennis.cmd --seconds 15 --out .\runs\ball-01

# Kiểm tra nhanh Kinect và model trong 5 giây, không mở cửa sổ.
.\.venv\Scripts\python.exe -B -m rgbd.live camera --seconds 5 --headless

# Chạy nhận diện 2D trên ảnh riêng.
.\.venv\Scripts\python.exe -B -m rgbd.live image .\ball.jpg
```

`--confidence 0.15` cho phép xem thêm các ứng viên điểm thấp. Model chạy mặc định với đầu vào 640 pixel. Có thể thử `--imgsz 960` hoặc `--imgsz 1280` khi bóng nhỏ, đồng thời theo dõi tốc độ xử lý. `--motion cv` là mặc định cho chuyển động gần đều. `--motion ballistic` dùng cho bóng đã rời tay, với trọng lực theo trục `-Y` sau khi xác nhận camera đặt cân bằng. Hệ thống không mô hình hóa va chạm hoặc nảy. Tọa độ đo từ bề mặt bóng mà Kinect nhìn thấy. Vận tốc và dự đoán xuất hiện sau ít nhất ba phép đo ghép vào cùng track. Track ngoại suy tối đa 0,30 giây khi thiếu phép đo.

Khi xử lý chậm hơn luồng Kinect, ứng dụng ưu tiên cặp RGB-D mới nhất và dùng timestamp thật để tính bước thời gian. `processed_fps` là tốc độ xử lý, còn `received_rgbd_pairs` là số cặp nhận từ cầu nối. Tỷ lệ depth của toàn ảnh và số khung có depth **trên bóng** là hai chỉ số riêng. File HTML của một phiên có chứa ảnh RGB cuối từ Kinect.

## Nguồn model và cài đặt lại

`models/tennis-ball.onnx` là YOLOv8n được chủ kho tinh chỉnh cho bóng tennis, dùng ONNX Runtime trên CPU. `setup.ps1` lấy trọng số từ revision `83ec86c` và kiểm tra SHA-256 `0020f0356923712947d223fcbbe053748ee862f5642c963041a6dc67ca8f910d`. Thẻ Hugging Face ghi Apache 2.0, còn metadata nhúng trong chính file ONNX ghi AGPL-3.0. Project dùng bản tải này để thử nghiệm cục bộ. [Thẻ model RJTPP](https://huggingface.co/RJTPP/tennis-ball-detection) · [Giấy phép được file ONNX dẫn chiếu](https://ultralytics.com/license)

Máy đã cài Kinect SDK v1.8. Khi chuyển máy, cần Windows x64, Kinect SDK v1.8 và Python 3.14 x64. Chạy `.\setup.ps1` để tạo môi trường Python riêng, tải model và biên dịch cầu nối. Nếu SDK báo đã kết nối nhưng mở luồng bị `0x80004005`, chạy từ cửa sổ PowerShell thông thường hoặc mở `Start-Tennis.cmd` từ File Explorer. [Microsoft Kinect SDK v1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278)

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
