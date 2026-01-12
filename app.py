import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tempfile
import os
import matplotlib.pyplot as plt
import io

# サーバー環境でのエラー（Tcl/Tkエラー）を避けるための設定
plt.switch_backend('Agg')
plt.rcParams['mathtext.fontset'] = 'cm'

# --- グラフ描画関数 ---
def create_graph_image(df_sub, x_col, y_col, x_label, y_label, x_unit, y_unit, color, size):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    try:
        if not df_sub.empty:
            ax.plot(df_sub[x_col], df_sub[y_col], color=color, linewidth=2)
            ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=50)
        
        ax.set_title(f"{y_label} - {x_label}", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"{x_label} [{x_unit}]")
        ax.set_ylabel(f"{y_label} [{y_unit}]")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
        return cv2.resize(img, (size, size))
    except:
        plt.close(fig)
        return np.zeros((size, size, 3), dtype=np.uint8)

st.set_page_config(page_title="CartGrapher Studio", layout="wide")
st.title("🚀 CartGrapher Studio (Basic)")

# --- サイドバー：設定 ---
st.sidebar.header("Kinema-Cart 設定")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mass = st.sidebar.number_input("台車の質量 m (kg)", value=0.100, format="%.3f")
mask_size = st.sidebar.slider("解析エリア半径 (px)", 50, 400, 200, 10)

# 色の定義（緑とピンク）
LOWER_GREEN = (np.array([35, 50, 50]), np.array([85, 255, 255]))
LOWER_PINK = (np.array([140, 40, 40]), np.array([180, 255, 255]))

uploaded_file = st.file_uploader("Kinema-Cartの動画を選択", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    status = st.empty()
    progress_bar = st.progress(0.0)
    data_log = []
    total_angle, prev_angle = 0.0, None
    gx, gy = np.nan, np.nan
    
    # --- 解析ループ ---
    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 中心(緑)
        mask_g = cv2.inRange(hsv, LOWER_GREEN[0], LOWER_GREEN[1])
        con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if con_g:
            c = max(con_g, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0: gx, gy = M["m10"]/M["m00"], M["m01"]/M["m00"]

        # 外周(ピンク)
        bx, by = np.nan, np.nan
        if not np.isnan(gx):
            m_circle = np.zeros((h_orig, w_orig), dtype=np.uint8)
            cv2.circle(m_circle, (int(gx), int(gy)), mask_size, 255, -1)
            mask_p = cv2.inRange(cv2.bitwise_and(hsv, hsv, mask=m_circle), LOWER_PINK[0], LOWER_PINK[1])
            con_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if con_p:
                cp = max(con_p, key=cv2.contourArea)
                Mp = cv2.moments(cp)
                if Mp["m00"] != 0: bx, by = Mp["m10"]/Mp["m00"], Mp["m01"]/Mp["m00"]

        if not np.isnan(gx) and not np.isnan(bx):
            curr_a = np.arctan2(by - gy, bx - gx)
            if prev_angle is not None:
                diff = curr_a - prev_angle
                if diff > np.pi: diff -= 2 * np.pi
                if diff < -np.pi: diff += 2 * np.pi
                total_angle += diff 
            prev_angle = curr_a

        data_log.append({"t": f_idx/fps, "x": total_angle*(radius_cm/100.0)})
        if f_idx % 20 == 0: progress_bar.progress(f_idx/total_frames)

    cap.release()
    df = pd.DataFrame(data_log).interpolate().ffill().bfill().fillna(0.0)
    
    # 物理量計算
    if len(df) > 10:
        df["x"] = savgol_filter(df["x"], 11, 2)
        df["v"] = savgol_filter(df["x"].diff().fillna(0)*fps, 21, 2)
        df["a"] = savgol_filter(df["v"].diff().fillna(0)*fps, 21, 2)
        df["F"] = float(mass) * df["a"]

    # グラフ表示
    st.subheader("📊 解析結果")
    ps = 450
    c1, c2 = st.columns(2)
    with c1: st.image(create_graph_image(df, "t", "x", "t", "x", "s", "m", "blue", ps), channels="BGR")
    with c2: st.image(create_graph_image(df, "t", "v", "t", "v", "s", "m/s", "red", ps), channels="BGR")
    with c1: st.image(create_graph_image(df, "t", "a", "t", "a", "s", "m/s^2", "green", ps), channels="BGR")
    with c2: st.image(create_graph_image(df, "x", "F", "x", "F", "m", "N", "purple", ps), channels="BGR")

    # CSV保存のみ残す
    st.download_button("📊 CSV保存", df.to_csv(index=False).encode('utf_8_sig'), "kinema_cart_data.csv")
    os.remove(tfile.name)
