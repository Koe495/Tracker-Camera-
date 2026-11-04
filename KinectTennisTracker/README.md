# Kinect RGB-D Tracker

Project Windows tại `E:\KinectTennisTracker`. Kinect for Windows v1 model 1517 thu RGB và depth. Chọn YOLOv8n nhận diện bóng tennis hoặc YOLO11s nhận diện UAV. Cả hai chế độ dùng tracker Kalman và các điểm depth đã căn theo RGB để ước lượng XYZ, vận tốc, hướng và vị trí dự kiến từ 0,1 đến 0,5 giây.

## Chạy trực tiếp

Kết nối Kinect với nguồn và USB. Mở `Start-Tracker.cmd` trong thư mục project, nhấn **T** cho bóng tennis hoặc **U** cho UAV. Nhấn **Q** hoặc **Esc** để dừng phiên rồi chọn chế độ khác. `Start-Tennis.cmd` vẫn mở chế độ tennis mặc định.

- **RGB** hiển thị ảnh màu và hộp nhận diện bóng. Hộp xanh có phép đo XYZ, hộp vàng chỉ có nhận diện 2D.
- **Depth** hiển thị độ sâu đã căn theo RGB của cùng cặp khung. Màu chuyển từ gần 0,8 m đến xa 4,0 m, pixel tối biểu thị depth thiếu hoặc ngoài khoảng đo. Hộp quanh bóng ghi khoảng cách Z khi có phép đo hợp lệ.
- **3D Motion Prediction** hiển thị trục X sang phải, Y hướng lên và Z hướng ra xa cảm biến. Mũi tên biểu thị hướng dịch chuyển hiện tại. Chấm lớn là vị trí hiện tại và các chấm nhỏ là vị trí dự báo cách nhau 0,1 giây. Vòng rỗng lớn đánh dấu track đang ngoại suy khi mất đo.
- **RGB / Forecast Boxes** dùng ảnh RGB hiện tại làm nền. Hộp nét liền là phát hiện hiện tại, còn hộp nét đứt là dự báo sau 0,1–0,5 giây. Chuyển động X/Y ở cùng độ sâu chủ yếu làm hộp dịch trong ảnh. Chuyển động Z làm hộp lớn khi đến gần và nhỏ khi đi xa. Hộp cũng có thể dịch tâm khi vật lệch trục quang học.
- **Tín hiệu chiều sâu trong khung dự báo.** Hộp màu cam khi tiến gần, xanh lơ khi ra xa, xám khi độ thay đổi Z trong khoảng ±3 cm. Dòng `dZ (m) vs now` hiện giá trị có dấu ở cả năm mốc cho track ghi ID, kể cả khi hộp chồng nhau hoặc ra ngoài ảnh. `Box +0.5s: x…` là hệ số kích thước so với hộp nhận diện gần nhất. Khi có nhiều track, dòng số ưu tiên ID nhỏ nhất trong các track có phép đo hiện tại.
- **Độ đậm của hộp dự báo.** Nét đứt rõ dần theo tốc độ trung bình dọc Z ở từng mốc, tính bằng `|ΔZ| / thời gian dự báo`. Độ đậm tăng tuyến tính từ 35% ở 0 m/s đến 95% ở 2 m/s và giữ mức tối đa trên ngưỡng này. Dấu của ΔZ vẫn quyết định màu.
- Lần thử đầu nên dùng một bóng, giữ yên ở khoảng 1–1,5 m rồi dịch ngang từ từ. Để bóng tách rõ khỏi người cầm và khỏi tường phía sau. Cảnh rộng có thể có nhiều pixel depth trống mà vùng bóng vẫn đủ điểm.

```powershell
Set-Location E:\KinectTennisTracker
.\Start-Tracker.cmd
```

Có thể chọn model trực tiếp từ PowerShell. Mỗi phiên khởi tạo bộ theo dõi riêng cho loại vật thể đã chọn.

```powershell
.\.venv\Scripts\python.exe -B -m rgbd.live camera --detector uav
.\.venv\Scripts\python.exe -B -m rgbd.live camera --detector tennis
```

## Ghi kết quả hoặc thử riêng model

```powershell
# Phiên 15 giây, tạo dashboard 3D phát lại, CSV và thống kê JSON.
.\Start-Tennis.cmd --seconds 15 --out .\runs\ball-01

# Kiểm tra nhanh Kinect và model trong 5 giây, không mở cửa sổ.
.\.venv\Scripts\python.exe -B -m rgbd.live camera --seconds 5 --headless

# Chạy nhận diện 2D trên ảnh riêng.
.\.venv\Scripts\python.exe -B -m rgbd.live image .\ball.jpg --detector tennis
.\.venv\Scripts\python.exe -B -m rgbd.live image .\uav.jpg --detector uav

# Phát lại bản ghi Kinect với model UAV.
.\.venv\Scripts\python.exe -B -m rgbd replay .\recordings\uav.rgbd --detector uav --out .\runs\uav-replay
```

`--confidence 0.15` cho phép xem thêm các ứng viên điểm thấp. Model chạy mặc định với đầu vào 640 pixel. Có thể thử `--imgsz 960` hoặc `--imgsz 1280` khi bóng nhỏ, đồng thời theo dõi tốc độ xử lý. `--motion cv` là mặc định cho chuyển động gần đều. `--motion ballistic` dùng cho bóng đã rời tay, với trọng lực theo trục `-Y` sau khi xác nhận camera đặt cân bằng. Hệ thống không mô hình hóa va chạm hoặc nảy. Tọa độ đo từ bề mặt bóng mà Kinect nhìn thấy. Vận tốc và dự đoán xuất hiện sau ít nhất ba phép đo ghép vào cùng track. Track ngoại suy tối đa 0,30 giây khi thiếu phép đo.

Khi xử lý chậm hơn luồng Kinect, ứng dụng ưu tiên cặp RGB-D mới nhất và dùng timestamp thật để tính bước thời gian. `processed_fps` là tốc độ xử lý, còn `received_rgbd_pairs` là số cặp nhận từ cầu nối. Tỷ lệ depth của toàn ảnh và số khung có depth **trên bóng** là hai chỉ số riêng. File HTML của một phiên có chứa ảnh RGB cuối từ Kinect.

## Nguồn model và cài đặt lại

`models/tennis-ball.onnx` là YOLOv8n được chủ kho tinh chỉnh cho bóng tennis, dùng ONNX Runtime trên CPU. `setup.ps1` lấy trọng số từ revision `83ec86c` và kiểm tra SHA-256 `0020f0356923712947d223fcbbe053748ee862f5642c963041a6dc67ca8f910d`. Thẻ Hugging Face ghi Apache 2.0, còn metadata nhúng trong chính file ONNX ghi AGPL-3.0. Project dùng bản tải này để thử nghiệm cục bộ. [Thẻ model RJTPP](https://huggingface.co/RJTPP/tennis-ball-detection) · [Giấy phép được file ONNX dẫn chiếu](https://ultralytics.com/license)

`models/uav.onnx` được xuất từ checkpoint YOLO11s một lớp `drone` trong `UAVDetection/best.pt`. Cả hai model chạy bằng ONNX Runtime trong môi trường Python của project. Khi thay checkpoint UAV, xuất lại ONNX từ môi trường có Ultralytics rồi dùng `--detector uav --weights <đường-dẫn.onnx>` để thử model mới. Định dạng đầu ra được hỗ trợ là mô hình YOLO detect một lớp xuất với `nms=False`. [Hướng dẫn xuất model Ultralytics](https://docs.ultralytics.com/modes/export)

```powershell
yolo export model=".\UAVDetection\best.pt" format=onnx imgsz=640 dynamic=True simplify=False opset=17 nms=False device=cpu
.\.venv\Scripts\python.exe -B -m rgbd.live camera --detector uav --weights .\UAVDetection\best.onnx
```

Video trong `UAVDetection/Testing` dùng để kiểm tra nhận diện RGB. Để kiểm tra tọa độ và dự báo 3D của UAV, cần UAV hoặc vật thể mẫu thật trong vùng Kinect có depth hợp lệ. Hai chỉ số `frames_with_2d_target` và `frames_with_3d_target` trong báo cáo giúp tách hai phép thử.

Máy đã cài Kinect SDK v1.8. Khi chuyển máy, cần Windows x64, Kinect SDK v1.8 và Python 3.14 x64. Chạy `.\setup.ps1` để tạo môi trường Python riêng, tải model và biên dịch cầu nối. Nếu SDK báo đã kết nối nhưng mở luồng bị `0x80004005`, chạy từ cửa sổ PowerShell thông thường hoặc mở `Start-Tennis.cmd` từ File Explorer. [Microsoft Kinect SDK v1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278)

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
