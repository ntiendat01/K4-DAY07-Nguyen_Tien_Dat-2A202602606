# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Tiến Đạt
**Nhóm:** \[Tên nhóm]
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**

> Độ tương tự cosine cao nghĩa là hai vector embedding có hướng gần giống nhau trong không gian vector. Với văn bản, điều này thường cho thấy hai đoạn có nội dung, chủ đề hoặc ý nghĩa gần nhau, dù cách diễn đạt có thể khác.

**Ví dụ có độ tương tự CAO:**

- Câu A: Ký túc xá Bách Khoa có 435 phòng ở và phục vụ sinh viên nội trú.
- Câu B: Sinh viên nội trú tại Đại học Bách khoa Hà Nội được bố trí trong khu ký túc xá nhiều phòng.
- Tại sao tương đồng: Cả hai câu cùng nói về ký túc xá Bách Khoa, phòng ở và sinh viên nội trú.

**Ví dụ có độ tương tự THẤP:**

- Câu A: Sinh viên đăng ký ký túc xá theo từng học kỳ.
- Câu B: Thư viện cho phép mượn sách giáo trình trong 10 ngày.
- Tại sao khác: Hai câu nói về hai dịch vụ khác nhau, một câu về ký túc xá và một câu về thư viện.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**

> Cosine similarity tập trung vào hướng của vector nên phù hợp để so sánh ý nghĩa, ít bị ảnh hưởng bởi độ dài vector. Với text embeddings, hai đoạn cùng chủ đề thường có hướng gần nhau, còn khoảng cách Euclid có thể bị nhiễu bởi độ lớn vector.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk\_size=500, overlap=50. Bao nhiêu chunks?**

> Bước nhảy giữa hai chunk là `chunk_size - overlap = 500 - 50 = 450`.
> Công thức: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`.
> Đáp án: 23 chunks.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**

> Khi overlap tăng lên 100, bước nhảy còn `500 - 100 = 400`, số chunk là `ceil((10000 - 100) / 400) = ceil(24.75) = 25`, tức nhiều hơn. Overlap lớn giúp giữ ngữ cảnh giữa hai chunk liên tiếp, nhưng đổi lại làm tăng số chunk cần lưu và tìm kiếm.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**SentenceChunker.chunk** — hướng tiếp cận:

> Tôi dùng regex `(?<=[.!?])(?:\s+|\n+)` để tách tại vị trí sau dấu kết thúc câu nhưng vẫn giữ lại dấu câu trong nội dung. Sau đó tôi strip khoảng trắng, bỏ câu rỗng và gom tối đa `max_sentences_per_chunk` câu thành một chunk. Trường hợp text rỗng trả về `[]`, còn `max_sentences_per_chunk` được ép tối thiểu là 1.

**RecursiveChunker.chunk / \_split** — hướng tiếp cận:

> RecursiveChunker thử tách văn bản theo thứ tự separator từ lớn đến nhỏ: đoạn, dòng, câu, khoảng trắng, rồi cắt cứng nếu không còn separator. Base case là text rỗng thì trả `[]`, text ngắn hơn `chunk_size` thì trả một chunk, còn hết separator thì cắt theo `chunk_size`. Sau khi tách, tôi gom các mảnh nhỏ liền nhau lại để tránh tạo quá nhiều chunk vụn.

### Lớp EmbeddingStore

**add\_documents + search** — hướng tiếp cận:

> Mỗi `Document` được chuẩn hóa thành record gồm `id`, `content`, `metadata` và `embedding`; metadata được copy để tránh sửa nhầm object của người gọi. Khi search, query được embed rồi so sánh với từng record bằng dot product, sau đó sắp xếp giảm dần theo score và trả về tối đa `top_k` kết quả.

**search\_with\_filter + delete\_document** — hướng tiếp cận:

> `search_with_filter` lọc metadata trước rồi mới chạy similarity search trên tập ứng viên còn lại, vì nếu search trước rồi lọc sau thì các slot top-k có thể bị tài liệu sai chiếm hết. `delete_document` xóa mọi record có `metadata["doc_id"]` khớp với doc\_id cần xóa và trả `True` nếu số lượng record giảm.

### Tác tử KnowledgeBaseAgent

**answer** — hướng tiếp cận:

> Agent gọi `store.search()` để lấy top-k chunk liên quan, sau đó dựng context có đánh số `[1]`, `[2]`, `[3]` kèm nguồn từ metadata. Prompt yêu cầu LLM chỉ trả lời dựa trên context được cung cấp, nếu không đủ thông tin thì nói không tìm thấy trong context, và trích dẫn số chunk khi trả lời.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
python -m unittest tests.test_solution -v

Ran 42 tests in 0.019s

OK
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A                                            | Câu B                                                    | Dự đoán | Điểm thực tế | Đúng? |
| --- | ------------------------------------------------ | -------------------------------------------------------- | ------- | ------------ | ----- |
| 1   | Ký túc xá có phòng ở cho sinh viên nội trú.      | Sinh viên có thể ở trong khu nội trú của trường.         | cao     | Cao          | Đúng  |
| 2   | Lệ phí nhà X1 là 2.700.000 đồng.                 | Nhà X2 có lệ phí 3.950.000 đồng.                         | cao     | Cao          | Đúng  |
| 3   | PTIT bố trí 460 chỗ ở tại KTX B2.                | KTX B1 có 40 chỗ ở cho sinh viên.                        | cao     | Cao          | Đúng  |
| 4   | Sinh viên đăng ký ký túc xá qua biểu mẫu online. | Trường công bố danh sách tuyển sinh đại học chính quy.   | thấp    | Thấp         | Đúng  |
| 5   | Mức giá điện nước có file đính kèm PDF.          | Sân bóng rổ phục vụ hoạt động thể thao trong khuôn viên. | thấp    | Thấp         | Đúng  |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

> Cặp 2 có thể gây bất ngờ vì hai câu nói về hai nhà khác nhau, nhưng vẫn tương đồng cao do cùng trường nghĩa về lệ phí ký túc xá. Điều này cho thấy embedding thường bắt được chủ đề và từ vựng chung, nhưng không phải lúc nào cũng phân biệt tốt chi tiết cụ thể như X1 hay X2 nếu không kiểm tra nội dung đáp án.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query)                                                                                                         | Top-1 Chunk truy xuất được (tóm tắt)                                         | Điểm Score | Có liên quan không? (Relevant)                 | Câu trả lời của Agent (tóm tắt)                                                                                              |
| - | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 1 | Ký túc xá Đại học Bách khoa Hà Nội có bao nhiêu dãy nhà, bao nhiêu phòng và có thể đón nhận khoảng bao nhiêu sinh viên? | `hust-dormitory-overview#0` — giới thiệu KTX Bách Khoa                       | 0.503      | Có liên quan, nhưng số liệu đầy đủ nằm ở top-2 | KTX Bách Khoa có 10 dãy nhà, 435 phòng, khoảng 4.200 sinh viên.                                                              |
| 2 | Cơ sở vật chất và lệ phí nhà X1, nhà X2 cho tân sinh viên K71 của HUCE là gì?                                           | `hust-dormitory-overview#1` — cơ sở vật chất KTX Bách Khoa                   | 0.346      | Không đúng trường; chunk đúng ở top-2          | Nhà X1 không có điều hòa/bình nước nóng, lệ phí 2.700.000 đồng; nhà X2 có điều hòa/bình nước nóng, lệ phí 3.950.000 đồng.    |
| 3 | PTIT bố trí bao nhiêu chỗ ở tại KTX B1, B2 và cơ sở Ngọc Trục cho sinh viên khóa 2025?                                  | `dorm-slot#0` — thông báo bố trí chỗ ở nội trú PTIT                          | 0.430      | Có liên quan; chi tiết số chỗ nằm trong top-3  | B1 có 40 chỗ, B2 có 460 chỗ, Ngọc Trục có 340 chỗ.                                                                           |
| 4 | Sinh viên Đại học Thương mại đăng ký ở Ký túc xá cơ sở Hà Nội theo quy trình nào và thời hạn đăng ký là khi nào?        | `tmu-dormitory-registration-hanoi#0` — thông báo cách thức đăng ký KTX       | 0.706      | Có                                             | Thời hạn 25/08/2023-30/08/2023; đăng ký qua biểu mẫu, xét đúng đối tượng ưu tiên và nhận tin nhắn xác nhận nếu đủ điều kiện. |
| 5 | Mức giá điện nước tại khu nội trú được ban hành ngày nào và file đính kèm tên gì?                                       | `tmu-dormitory-electric-water-fees#0` — văn bản điều chỉnh mức giá điện nước | 0.546      | Có                                             | Ngày ban hành 05/08/2024; file đính kèm `dieu-chinh-gia-dien-nuoc-kntpdf-1727328575.pdf`.                                    |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

Chi tiết top-3, điểm naive và điểm content-aware được lưu trong `ket_qua_benchmark.txt`. Backend đo là `LexicalHashEmbedder`, một baseline từ khóa cục bộ và deterministic, chưa phải embedding ngữ nghĩa thật nên kết quả chủ yếu phản ánh mức trùng từ khóa và độ mạch lạc của chunk.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**

> Chỉ kiểm tra `doc_id` trong top-3 là chưa đủ vì chunk đúng tài liệu có thể không chứa đáp án. Cần kiểm tra nội dung top-3 bằng các chuỗi đặc trưng như số tiền, ngày ban hành hoặc tên file đính kèm để biết retrieval có thật sự trả lời được câu hỏi không.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí                                        | Điểm tự đánh giá |
| ----------------------------------------------- | ---------------- |
| Khởi động (Warm-up)                             | 5 / 5            |
| Hướng tiếp cận của tôi (My Approach)            | 10 / 10          |
| Hoàn thiện code (Core Implementation — tests)   | 30 / 30          |
| Dự đoán độ tương tự (Similarity Predictions)    | 5 / 5            |
| Kết quả truy xuất của tôi (Competition Results) | 9 / 10           |
| **Tổng phần cá nhân**                           | **59 / 60**      |
