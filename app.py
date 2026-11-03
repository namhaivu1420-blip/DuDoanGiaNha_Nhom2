import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Dự đoán giá nhà AI", page_icon="🏠", layout="wide")
FEATURES = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]

@st.cache_data(show_spinner=False)
def create_data(n=1000):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "OverallQual": rng.integers(1, 11, n),
        "GrLivArea": rng.integers(500, 4501, n),
        "GarageCars": rng.integers(0, 5, n),
        "TotalBsmtSF": rng.integers(0, 2501, n),
        "YearBuilt": rng.integers(1950, 2024, n),
    })
    df["SalePrice"] = (
        df["OverallQual"] * 25000 + df["GrLivArea"] * 85
        + df["GarageCars"] * 15000 + df["TotalBsmtSF"] * 40
        + (df["YearBuilt"] - 1950) * 500 + rng.normal(0, 15000, n)
    )
    return df

@st.cache_resource(show_spinner=False)
def train_models():
    df = create_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["SalePrice"], test_size=0.2, random_state=42
    )
    # Both models use the same split. Pipeline fits scaling on training data only.
    ann = Pipeline([
        ("x_scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                               solver="adam", max_iter=1200, early_stopping=True,
                               n_iter_no_change=30, random_state=42)),
    ])
    # Scale the target to make both optimization tasks numerically stable.
    from sklearn.compose import TransformedTargetRegressor
    ann = TransformedTargetRegressor(regressor=ann, transformer=StandardScaler())
    svr = TransformedTargetRegressor(
        regressor=Pipeline([
            ("x_scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=100, epsilon=0.1, gamma="scale")),
        ]), transformer=StandardScaler()
    )
    ann.fit(X_train, y_train)
    svr.fit(X_train, y_train)
    preds = {"ANN": ann.predict(X_test), "SVM (SVR)": svr.predict(X_test)}
    scores = pd.DataFrame({name: {
        "MAE (USD)": mean_absolute_error(y_test, p),
        "MSE (USD²)": mean_squared_error(y_test, p),
        "RMSE (USD)": np.sqrt(mean_squared_error(y_test, p)),
        "R²": r2_score(y_test, p),
    } for name, p in preds.items()}).T
    comparison = pd.DataFrame({"Giá thực tế": y_test.to_numpy(),
                               "ANN": preds["ANN"], "SVM (SVR)": preds["SVM (SVR)"]})
    return ann, svr, scores, comparison, df

st.title("🏠 DỰ ĐOÁN GIÁ NHÀ BẰNG AI")
st.write("Ứng dụng so sánh mạng nơ-ron nhân tạo **ANN** và máy véc-tơ hỗ trợ hồi quy **SVM (SVR)**.")
st.warning("⚠️ Dữ liệu 1.000 căn nhà được **mô phỏng** để học tập. Kết quả không phải định giá thị trường hay kết quả trên dữ liệu Kaggle.")
with st.spinner("Đang huấn luyện mô hình lần đầu, vui lòng đợi..."):
    ann, svr, scores, comparison, df = train_models()

st.sidebar.header("⚙️ Thông số ngôi nhà")
with st.sidebar.form("house_form"):
    quality = st.slider("Chất lượng tổng thể (1–10)", 1, 10, 6)
    area = st.number_input("Diện tích sinh hoạt (sqft)", 500, 4500, 1500, step=50)
    garage = st.selectbox("Số ô tô trong gara", [0, 1, 2, 3, 4], index=2)
    basement = st.number_input("Diện tích tầng hầm (sqft)", 0, 2500, 800, step=50)
    year = st.slider("Năm xây dựng", 1950, 2023, 2005)
    submitted = st.form_submit_button("🚀 Dự đoán giá nhà", use_container_width=True)

prediction_tab, evaluation_tab, data_tab = st.tabs(["🏡 Dự đoán", "📊 Đánh giá mô hình", "📁 Dữ liệu"])
with prediction_tab:
    if submitted:
        sample = pd.DataFrame([{
            "OverallQual": quality, "GrLivArea": area, "GarageCars": garage,
            "TotalBsmtSF": basement, "YearBuilt": year,
        }], columns=FEATURES)
        pred_ann = float(ann.predict(sample)[0])
        pred_svr = float(svr.predict(sample)[0])
        c1, c2 = st.columns(2)
        c1.metric("🤖 ANN", f"${pred_ann:,.0f}")
        c2.metric("⚙️ SVM (SVR)", f"${pred_svr:,.0f}")
        st.info(f"Trung bình hai dự đoán: **${(pred_ann + pred_svr) / 2:,.0f}** (không phải mô hình đã được kiểm định riêng).")
        chart = pd.DataFrame({"Mô hình": ["ANN", "SVM (SVR)"],
                              "Giá dự đoán (USD)": [pred_ann, pred_svr]}).set_index("Mô hình")
        st.bar_chart(chart)
    else:
        st.info("👈 Nhập thông số bên trái rồi chọn **Dự đoán giá nhà**.")

with evaluation_tab:
    st.subheader("Đánh giá trên cùng tập kiểm tra (20%)")
    st.dataframe(scores.style.format({"MAE (USD)": "{:,.2f}", "MSE (USD²)": "{:,.2f}",
                                      "RMSE (USD)": "{:,.2f}", "R²": "{:.4f}"}),
                 use_container_width=True)
    best = scores["RMSE (USD)"].idxmin()
    st.success(f"Mô hình có RMSE thấp hơn trên tập kiểm tra: **{best}**.")
    st.caption("MAE, MSE, RMSE càng thấp càng tốt; R² càng cao càng tốt. Đây là bài toán hồi quy nên không dùng Accuracy, Precision, Recall, F1-score.")
    st.subheader("Giá thực tế và giá dự đoán (50 mẫu đầu)")
    st.line_chart(comparison.head(50).reset_index(drop=True))

with data_tab:
    st.write("Dữ liệu mô phỏng: 1.000 mẫu, 5 đặc trưng đầu vào, 1 giá mục tiêu.")
    st.dataframe(df.head(20), use_container_width=True)
    st.download_button("⬇️ Tải dữ liệu CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="du_lieu_gia_nha_mo_phong.csv", mime="text/csv")
