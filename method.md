# Báo cáo Phương pháp luận: Đăng ký ảnh Y tế 3D sử dụng SynthMorph

Tài liệu này hệ thống lại các khái niệm, cơ sở toán học và phương pháp thiết kế cấu trúc kiến trúc của mô hình Đăng ký ảnh 3D, dựa trên nền tảng phương pháp SynthMorph. 

---

## 1. Giới thiệu Bài toán Đăng ký ảnh (Image Registration)

Trong y tế, một bệnh nhân có thể được chụp nhiều ảnh quét 3D ở các thời điểm hoặc bằng các thiết bị khác nhau (ví dụ: MRI T1w, T2w, chụp cắt lớp CT). Các ảnh này thường không đồng nhất về vị trí và hình dạng do sự xê dịch cơ học hoặc sinh lý học (nhịp thở, cử động).
**Đăng ký ảnh (Registration)** là bài toán tìm kiếm một trường biến dạng không gian (Deformation Field) nhằm dịch chuyển vị trí các điểm ảnh của **Ảnh di động (Moving Image)** sao cho nó trùng khớp hoàn toàn về mặt giải phẫu với **Ảnh tham chiếu (Fixed Image)**.

---

## 2. Giải pháp SynthMorph và Tính "Bất biến độ tương phản"

Các mô hình học sâu truyền thống thường nội suy độ lệch trực tiếp từ mức xám (intensity) của các cặp ảnh y khoa thực tế. Điểm yếu của phương pháp này là mạng chỉ ghi nhớ đặc trưng cường độ quang học của loại ảnh huấn luyện, dẫn đến việc thiếu sự chính xác khi nhận diện các ảnh đa thể thức (Cross-modality), ví dụ đăng ký ảnh MRI với CT cứng.

**SynthMorph** giải quyết điểm yếu này bằng phương pháp học "bất biến độ tương phản" (Contrast-invariant):
1. Phương pháp **không sử dụng ảnh y tế thật** trong huấn luyện.
2. Mô hình tiến hành sinh các hệ cấu trúc không gian hình học (Label Maps).
3. Đổ các dải màu (mức xám) ngẫu nhiên theo phân phối xác suất lên các lưới hình học trên để tạo thành cặp ảnh $m$ và $f$.
4. Thông qua việc bắt cặp, mạng nơ-ron buộc phải tối ưu hóa dựa trên các ranh giới cấu trúc giải phẫu (Topology / Edges) thay vì ghi nhớ màu sắc mức xám, từ đó cho phép áp dụng trên nhiều thể thức (modalities) khác nhau.

---

## 3. Kiến trúc Mô hình (Model Architecture)

### 3.1. Mạng U-Net 3D
U-Net đóng vai trò trích xuất đặc trưng hình ảnh và dự đoán thông số biến dạng. Mạng nhận đầu vào là cặp ảnh 3D $(m, f)$ được ghép kênh $\rightarrow (B, 2, D, H, W)$.
- **Encoder:** Nén không gian qua 4 mức phân giải ($D/2, D/4, D/8, D/16$) sử dụng lớp tích chập `Conv3D` cấu hình `LeakyReLU`, nhằm trích xuất các đặc trưng không gian ở quy mô lớn.
- **Decoder:** Khôi phục độ phân giải bằng lớp `Upsample` (nội suy `trilinear`), kết hợp với dữ liệu truyền ngang (Skip-connections) từ Encoder để bảo toàn chi tiết biên cạnh.

### 3.2. Trường vận tốc tĩnh (Stationary Velocity Field - SVF)
Mô hình hình học vật lý chặt chẽ cấm việc xuất trực tiếp véctơ chuyển vị (Displacement field) do nguy cơ giao cắt lưới (Negative Jacobian), dẫn đến cấu trúc mô bị biến tướng và đứt gãy.
Thay vào đó, U-Net chỉ tính ra một **Trường vận tốc tĩnh (SVF, ký hiệu $\mathbf{v}$)**. $\mathbf{v}$ này trơn tru và bảo đảm tính khả vi liên tục toán học toàn cục.

### 3.3. Tích phân Scaling and Squaring (Trường biến dạng Diffeomorphic)
Trường biến dạng $\phi$ cuối cùng được tính ra khỏi $\mathbf{v}$ thông qua phương trình vi phân (ODE):
$$ \frac{d\phi^{(t)}}{dt} = \mathbf{v}(\phi^{(t)}) $$
Để giải hệ phương trình vi phân này trên nền tảng máy tính, thuật toán **Scaling and Squaring** chia tỷ lệ vận tốc $\mathbf{v}$ đi $2^K$ lần ($K$ là bước chia, thường chọn $K=5$ hoặc $K=7$), sau đó dịch chuyển liên tục $K$ lần tích chập. Điều này sinh ra một trường biến đổi $\phi$ duy trì được hình học đồng phôi (Topology preservation/Diffeomorphic mapping).

### 3.4. Mạng biến đổi không gian (Spatial Transformer Network - STN)
$\phi$ tiếp tục được nạp vào hàm `F.grid_sample` nội suy không gian 3D mượt mà để bẻ cong ảnh $m$. STN giữ mạng học sâu luôn trong quá trình nối tiếp lan truyền ngược (End-to-End differentiable).

---

## 4. Đặc điểm Các Hàm Suy hao (Loss Functions)

### 4.1. Hàm mất mát Soft Dice (Trùng khớp hình học - $\mathcal{L}_{dice}$)
Trong quá trình huấn luyện, mô hình chấm điểm sự trùng khớp thông qua nhãn giải phẫu (Label maps: $s_m$ và $s_f$) tương ứng của 2 ảnh $m$ và $f$.
Công thức Soft Dice như sau:
$$ \mathcal{L}_{dice} = 1 - \frac{1}{J} \sum_{j=1}^{J} \frac{2 \sum (s_m^{(j)} \circ \phi) \cdot s_f^{(j)} + \epsilon}{\sum (s_m^{(j)} \circ \phi) + \sum s_f^{(j)} + \epsilon} $$
**(Giải pháp Đạo hàm mạng):** Do $s_m$ là tọa độ nguyên rời rạc (Integers), việc nội suy gần nhất (Nearest Interpolation) để dịch chuyển lưới sẽ khiến hàm vi phân Gradient bằng $\mathbf{Zero}$ trên toàn không gian. Để mô hình có thể hội tụ, $s_m$ được chuyển hóa thành dạng xác suất (One-hot vectors) và sử dụng thuật toán nội suy **Bilinear**. Đạo hàm từ đó được đảm bảo thông suốt trong suốt cấu trúc lan truyền ngược (Back-propagation).

### 4.2. Hàm điều chuẩn độ trơn lưới ($\mathcal{L}_{reg}$)
Hình phạt đối với tình trạng bẻ cong lưới quá gắt nhằm cực tiểu hóa Soft Dice.
$$ \mathcal{L}_{reg} = \frac{1}{|V|} \sum_{x \in V} ||\nabla \mathbf{u}(x)||^2 $$
Hàm tính toán đạo hàm Gradient liên kề trên trường rời $\mathbf{u} = \phi - Id$. Kết hợp qua thông số $\lambda$, nó ép luồng biến dạng kéo giãn như một màng nilon mềm và tự nhiên.

---

## 5. Quy trình Bộ sinh Dữ liệu Giả Lập (Data Generators)

Data Generator là thành phần thay thế thế giới thực nhằm huấn luyện tập SynthMorph. Trong khuôn khổ triển khai này, có 2 biến thể Generator.

### 5.1. Bộ sinh Baseline (Variant A: sm-shapes)
Sử dụng phương pháp trong bài báo gốc.
- Sinh các hệ lưới nhiễu Gauss độc lập trên không gian độ phân giải thấp (VD: $10 \times 12 \times 14$).
- Upscale thông qua nội suy cơ bản.
- Kết hợp với lọc biên `AvgPool3D` liên tiếp nhiều vòng (Xấp xỉ hàm Gaussian Blur Smoothing) để tập hợp khối lại thành các kết cấu mô cơ bản tròn trịa (Blobby structures).
Nhược điểm của biến thể này là việc tính toán đa nhãn cực kì tiêu tốn bộ nhớ (>17GB VRAM) khi có $>20$ labels song song.

### 5.2. Biến thể Custom Generator (Variant B: Geometric-focused)
Biến thể này được thiết kế nhằm khắc phục mặt hạn chế về RAM cũng như tăng khả năng hiểu biết lâm sàng cho mạng.

**a. Cơ chế xử lý bộ nhớ Streaming Argmax:** 
Nhằm đưa việc huấn luyện mạng có thể thực thi tốt trên GPU dung lượng thấp (VD Tesla T4, RTX 3050). Việc lưu giữ đồng thời các bản ghi $(B, 26, D, H, W)$ bị bãi bỏ. Thuật toán sẽ tính cục diện tuần tự cho các nhãn (Streaming). Cứ khởi tạo xong 1 Nhãn và so sánh Argmax, bộ nhớ cache liền lập tức bị xóa bỏ (Garabage Collection mechanism). VRAM đòi hỏi giảm xuống $< 500MB$.

**b. Đội hình Khối Hình học Cứng (Geometric Primitives):** 
Mô phỏng hình hài giải phẫu đa dạng, mạng được tiếp xúc với viền cấu trúc sắc nét cắt cạnh (Sharp Edges):
- Hình cầu (Spheres), Hình nón bầu dục (Ellipsoids).
- Khối chữ nhật không gian và xoay vát (Cuboids, Rotated Cuboids).
- Ống trụ rỗng (Cylinders) tương tự đường khí quản và mao mạch.

**c. Chèn đa cấp độ Không Gian (Biomedical Deformations):**
Cảnh quan y khoa được áp lên 3 cấp vĩ mô (Tầm toàn diện, Phân khu, và Tổ chức vi mô), bổ trợ cùng các điều kiện biến vị có tính bệnh lý:
- **Twist (Hiện Tượng Xoắn Vặn):** Quay trường vectơ không gian theo hệ số tăng dần.
- **Inflate / Deflate (Phình Tụ Máng):** Thuật toán sử dụng lực phân cực xuyên tâm Exponential nhằm fake khối mô sụp võng hoặc khối u nội nang cấp (Tumor Growth).
- **Crumpled Fold:** Nhiễu cấp biên mô phỏng cuộn não.

### 5.3. Mô phỏng đặc tính Cường độ máy Quét MRI (GMM Synthesis)
Cuối cùng, phổ cấu xạ vật lý ánh sáng (Intensity) sẽ được ánh xạ vào nhãn. Quá trình áp dụng kỹ thuật ngẫu nhiên **Mô hình Trộn Gaussian - GMM (Gaussian Mixture Models)**, kết tủa cùng ba khuyết điểm vật lý phổ biến của chụp cắt lớp:
- Khuyết điểm nhòe biên (Partial Volume Effects - PVE).
- Nhiễu trường nền phân bổ lệch (Bias field artifacts).
- Căn chỉnh biến thiên dải nhòe tương phản hình ảnh (Gamma Warp Correction).
Các mẫu ảnh xuất ra được nạp vào U-Net hình thành hàng chục ngàn kịch bản học tập đối chiếu lý tưởng.
