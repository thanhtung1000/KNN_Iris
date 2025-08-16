import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------
# 1. Hàm tính khoảng cách Euclidean
# ----------------------------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# ----------------------------
# 2. Cài đặt KNN thủ công
# ----------------------------
class MyKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict_with_confidence(self, X_test):
        """Trả về (nhãn dự đoán, độ tin cậy)"""
        results = []
        for x in np.array(X_test):
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]
            count = Counter(k_labels)
            predicted_label, votes = count.most_common(1)[0]
            confidence = votes / self.k
            results.append((predicted_label, confidence))
        return results

    def predict(self, X_test):
        """Chỉ trả về nhãn dự đoán"""
        return [label for label, _ in self.predict_with_confidence(X_test)]

# ----------------------------
# 3. Load dữ liệu
# ----------------------------
df = pd.read_csv("Iris.csv")
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# ----------------------------
# 4. Train/Test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------
# 5. Huấn luyện mô hình
# ----------------------------
k = 5
knn = MyKNN(k=k)
knn.fit(X_train, y_train)

# ----------------------------
# 6. Giao diện Streamlit
# ----------------------------
st.title("🌸 Dự đoán loài hoa Iris với K‑NN (Custom)")
st.write("Nhập thông số của hoa để dự đoán")

# Ô nhập liệu
sl = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sw = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
pl = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
pw = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Nút dự đoán
if st.button("🔍 Dự đoán"):
    sample = [[sl, sw, pl, pw]]
    pred_label, conf = knn.predict_with_confidence(sample)[0]
    st.success(f"🌼 Loài hoa dự đoán: **{pred_label}** "
               f"(Độ tin cậy: {conf*100:.2f}%)")

# ----------------------------
# 7. Đánh giá mô hình
# ----------------------------
if st.checkbox("📊 Xem đánh giá mô hình"):
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.write(f"🎯 Độ chính xác: **{acc*100:.2f}%**")
    st.write("📊 Ma trận nhầm lẫn:")
    st.write(cm)
    st.write("📄 Báo cáo phân loại:")
    st.text(report)