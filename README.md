Ứng dụng Dự đoán Loài Hoa Iris với K‑NN (Custom) trên Web
Giới thiệu
Ứng dụng web này được xây dựng bằng [Streamlit](https://streamlit.io/) và thuật toán **K‑Nearest Neighbors (K‑NN)** tự cài đặt bằng Python.  
Người dùng có thể nhập **4 thông số đo** của hoa Iris và nhận dự đoán loài hoa cùng **tỉ lệ % độ tin cậy**.

Bộ dữ liệu sử dụng: **Iris Dataset** — gồm 150 mẫu, chia thành 3 lớp:
- *Iris setosa*
- *Iris versicolor*
- *Iris virginica*

---

##  Yêu cầu hệ thống
- Python >= 3.7
- Thư viện:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `scikit-learn`

Cài đặt nhanh toàn bộ:
```bash
pip install streamlit pandas numpy scikit-learn
RUN: streamlit run Iris_web.py
<img width="880" height="759" alt="image" src="https://github.com/user-attachments/assets/02894fe6-6dc5-420f-9b37-9f09c81ec241" />
<img width="817" height="601" alt="image" src="https://github.com/user-attachments/assets/43d675cb-fc02-4c28-bfc2-be7e62486ee4" />
