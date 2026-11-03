# Ứng dụng dự đoán giá nhà bằng ANN và SVM

Ứng dụng Streamlit sử dụng **MLPRegressor (ANN)** và **SVR (SVM)** để dự đoán giá nhà bằng USD trên 1.000 mẫu **giả lập**. Không sử dụng TensorFlow.

## Cấu trúc
- `app.py`: mã ứng dụng
- `requirements.txt`: thư viện cần cài đặt

## Triển khai Streamlit Community Cloud
1. Tải **cả hai file** `app.py` và `requirements.txt` vào **thư mục gốc** của cùng một nhánh GitHub.
2. Trong Streamlit Community Cloud, kiểm tra đúng repository, branch, và main file path `app.py`.
3. Chọn Python 3.12 khi tạo ứng dụng mới (nếu có tùy chọn).
4. Deploy. Mở **Manage app > Cloud logs** nếu báo lỗi.
5. Nếu vẫn có `No module named 'sklearn'`, xem **log cài đặt dependencies** và kiểm tra Streamlit có phát hiện đúng `requirements.txt` hay không. Nếu repository có `uv.lock`, `Pipfile`, `environment.yml` hoặc các file dependency khác, cần kiểm tra ưu tiên lựa chọn dependency của Streamlit Cloud.

## Chạy trên máy
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Lưu ý
- Train/test 80/20, chuẩn hóa trên train, dùng chung tập test.
- ANN và SVR là **mô hình hồi quy**; đánh giá bằng MAE, MSE, RMSE, R².
- Dữ liệu và kết quả **không phải** dữ liệu giá nhà Kaggle hay định giá thị trường thực tế.
