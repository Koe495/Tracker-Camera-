# Mốc tiếp theo: Kinect RGB–depth → đo vùng và phát hiện bề mặt gần

Triển khai theo lựa chọn dùng Kinect 1517 hiện có. Không thêm camera RGB thứ hai và không tính disparity stereo. Giữ `main.py` cũ để xem video; mở `depth_lab.py` cho chức năng mới hoặc chạy `python main.py --lab`.

## Chạy trong PyCharm

Chọn **Kinect 1517 - Measurement Live**, rồi Run. Chọn **Kinect 1517 - Measurement Demo** để thử dữ liệu giả. Môi trường `.venv-kinect` và requirements giữ nguyên.

```powershell
.\.venv-kinect\Scripts\python.exe depth_lab.py
.\.venv-kinect\Scripts\python.exe depth_lab.py --demo
```

## Đo vùng và kiểm tra căn chỉnh

1. Đặt hộp hoặc ghế trước tường, giữ Kinect cố định. Ảnh bên phải là **depth đã ánh xạ vào lưới pixel RGB**; ảnh depth gốc vẫn được giữ riêng để tính XYZ và vật cản.
2. Nhấn **O** để chồng màu depth lên RGB. Kiểm tra biên hộp ở giữa và hai mép ảnh. SDK dùng hiệu chuẩn của chính Kinect; không resize ảnh để giả lập căn chỉnh.
3. Nhấp chuột để chọn vùng nhỏ 11×11 hoặc kéo chuột trên một trong hai ảnh để chọn ROI lớn. Hai ảnh dùng cùng tọa độ RGB. Vùng không có dữ liệu vẫn để trống/đen, không lấp bằng depth của nền.
4. Xem **ROI median**, **valid** và **spatial std**. Median lấy các điểm hợp lệ; yêu cầu ít nhất 9 điểm và 50% ROI hợp lệ. Dữ liệu không đủ trả `INSUFFICIENT_DEPTH` và khoảng cách `N/A`.

Depth được lưu theo mm (giá trị Z của camera depth), không phải khoảng cách Euclid tới mọi pixel RGB. Khi ROI chứa cả vật và nền, median có thể đại diện nền; hãy chọn ROI nằm trong bề mặt cần đo. Spatial std lớn là dấu hiệu ROI chứa nhiều bề mặt hoặc nhiễu. Chênh timestamp tối đa 40 ms vẫn có thể tạo lệch khi vật chuyển động nhanh.

## Đánh giá bằng thước

Giữ một mặt phẳng trong ROI ở khoảng cách đã đo bằng thước từ cảm biến. Nhấn:

| Phím | Khoảng cách chuẩn |
|---|---|
| 1 | 1.000 mm |
| 2 | 1.500 mm |
| 3 | 2.000 mm |
| 4 | 3.000 mm |

Sau đó nhấn **E** để lấy 60 khung; ROI và khoảng cách chuẩn được khóa trong lần lấy mẫu đó. `measurements/eval_*.csv` chứa số đo từng khung, mã khung, timestamp, ROI, độ lệch với thước, tỷ lệ điểm hợp lệ; file JSON cùng tên chứa MAE, sai số có dấu trung bình và độ dao động median theo thời gian. Lần kết thúc sớm mang tiền tố `partial_`.

```powershell
python depth_lab.py --reference-mm 1500 --measure-frames 90
python depth_lab.py --demo --headless --frames 60 --evaluate --measure-frames 30 --preview lab_demo.png
```

Chỉ bật `--evaluate` khi đã đặt vật đúng khoảng cách chuẩn. Dữ liệu mô phỏng có `source=simulation`; không dùng làm số liệu thực nghiệm của camera. Không có số liệu thước thật được tạo sẵn.

## Phát hiện bề mặt gần / vật cản

Ba thẻ **LEFT / CENTER / RIGHT** dùng ảnh depth gốc và hướng tương đối theo góc nhìn camera. Phân tích dải 15%–85% chiều cao ảnh. Khối điểm gần phải đủ diện tích liên thông và xác nhận qua 3 cặp khung:

- `DANGER`: bề mặt trong khoảng đo hợp lệ và gần hơn hoặc bằng 1,0 m.
- `NEAR`: bề mặt gần hơn hoặc bằng 1,5 m.
- `CHECKING`: trạng thái đang thay đổi/chưa đủ ba khung xác nhận.
- `UNKNOWN`: dưới 30% vùng có depth hợp lệ. Không diễn giải điểm thiếu thành đường trống.
- `NO_NEAR_SURFACE`: chưa thấy cụm gần đạt tiêu chí trong vùng được quan sát; không mang nghĩa đường đi an toàn.

Để bỏ mặt sàn, hướng Kinect tương đối ngang và thấy đủ sàn ở nửa dưới, nhấn **F**. Hệ thống RANSAC trên XYZ hiệu chuẩn của SDK, chỉ nhận mặt phẳng gần nằm ngang và dưới camera 0,25–2,5 m. Nếu không đủ bằng chứng, từ chối hiệu chuẩn. Giữ nguyên vị trí/góc camera; sau khi dịch chuyển, nhấn **C** xóa mô hình cũ rồi **F** để hiệu chuẩn lại. Khi chưa hiệu chuẩn, mặt sàn vẫn có thể gây cảnh báo.

```powershell
python depth_lab.py --warning-mm 1800 --danger-mm 1100 --audio
python depth_lab.py --near-mode --min-depth 400 --max-depth 3000
```

`--audio` bật tiếng bíp ngắn qua một worker, tối đa một cảnh báo mỗi 3 giây; không chặn luồng lấy ảnh. Đây là cảnh báo bề mặt gần trong bản thử nghiệm, chưa nhận tên vật, chưa loại trừ được mọi loại nhiễu và không phát hiện được hố/bậc xuống. Cần đánh giá thực tế trước khi dùng cho tác vụ hỗ trợ di chuyển.

## Ghép module OCR và tìm vật của nhóm

`RGBDPacket` trong `perception.py` là hợp đồng dữ liệu:

| Trường | Ý nghĩa |
|---|---|
| frame_id | Mã khung ảnh màu |
| color_timestamp_ms / depth_timestamp_ms | Timestamp cảm biến, đơn vị ms |
| color_bgr | Ảnh BGR uint8 `(480,640,3)` để detector/OCR nhận trực tiếp |
| depth_aligned_mm | Ảnh uint16 `(480,640)` trong hệ tọa độ RGB; 0 không hợp lệ |
| source | `kinect1517` hoặc `simulation` |

Detector trả bounding box **`(x, y, width, height)`** theo ảnh RGB gốc và `frame_id`. Nếu detector resize/letterbox ảnh, phải chuyển box về ảnh 640×480 trước khi gọi:

```python
# packet là RGBDPacket tương ứng với khung đã đưa vào detector.
measurement = packet.measure_detection(detection_frame_id, (x, y, width, height))
if measurement.status == "OK":
    distance_m = measurement.median_mm / 1000
else:
    distance_m = None  # trả "không đo được" cho bộ điều phối
```

Hàm từ chối kết quả khác mã khung. Detector chậm cần giữ packet gốc trong bộ đệm riêng, không lấy box cũ đo với depth mới. OCR cũng nhận `packet.color_bgr`, trả text/vùng chữ/trạng thái về bộ điều phối; không tự mở một Kinect thứ hai. Chưa tích hợp YOLO/OCR thật vì mã của hai module chưa được cung cấp.

## Lưu dữ liệu và kiểm thử

Nhấn **S** lưu `.npz`: RGB, depth gốc, depth đã căn chỉnh, XYZ trong hệ camera depth (m), mặt phẳng sàn nếu có, metadata, ROI và kết quả ba vùng. Mở bằng `np.load(path, allow_pickle=False)` để xử lý lại offline. Snapshot không phải chuỗi video; thu chuỗi và tích hợp giọng nói OCR/detector là mốc tiếp theo.

```powershell
python -m unittest -v test_vision test_perception
```

Các kiểm tra gồm ABI SDK, giải mã depth, stride buffer, ánh xạ lệch tọa độ và lỗ trống, ROI thiếu dữ liệu, ghép khung, loại kết quả detector cũ, lọc điểm nhiễu, hiệu chuẩn sàn, xác nhận cảnh báo và CSV sai số. Kiểm tra live chỉ xác nhận nhận cặp khung và SDK mapper hoạt động; độ chính xác với thước và biên vật cần được người dùng thực nghiệm.
