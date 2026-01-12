import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tempfile
import os
import matplotlib.pyplot as plt
import io

# Matplotlib設定
plt.switch_backend('Agg')
plt.rcParams['mathtext.fontset'] = 'cm'

# --- グラフ生成関数 ---
def create_graph_image(df_sub, x_col, y_col, x_label, y_label, x_unit, y_unit, color, size, x_min, x_max, y_min, y_max):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    try:
        if not df_sub.empty:
            ax.plot(df_sub[x_col].values, df_sub[y_col].values, color=color, linewidth=2)
            ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=50)
        
        ax.set_title(f"${y_label}$ - ${x_label}$", fontsize=16, fontweight='bold')
        ax.set_xlabel(f"${x_label}$ [{x_unit}]", fontsize=14)
        ax.set_ylabel(f"${y_label}$ [{y_unit}]", fontsize=14)
        
        xr = max(float(x_max - x_min), 0.001)
        yr = max(float(y_max - y_min), 0.001)
        ax.set_xlim(float(x_min) - xr*0.05, float(x_max) + xr*0.05)
        ax.set_ylim(float(y_min) - yr*0.1, float(y_max) + yr*0.1)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        if x_col != 't': ax.axvline(0, color='black', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor='white')
        buf.seek(0)
        img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
        plt.close(fig)
        return cv2.resize(img, (size, size))
    except Exception:
        plt.close(fig)
        return np.zeros((size, size, 3), dtype=np.uint8)

st.set_page_config(page_title="CartGrapher Studio", layout="wide")
st.title("🚀 CartGrapher Studio")

# --- サイドバー設定 ---
st.sidebar.header("Kinema-Cart 設定")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mass = st.sidebar.number_input("台車の質量 $m$ (kg)", value=0.100, min_value=0.001, format="%.3f")
mask_size = st.sidebar.slider("解析エリア半径 (px)", 50, 400, 200, 10)

LOWER_GREEN = (np.array([35, 50, 50]), np.array([85, 255, 255]))
LOWER_PINK = (np.array([140, 40, 40]), np.array([180, 255, 255]))

uploaded_file = st.file_uploader("Kinema-Cartの実験動画を選択", type=["mp4", "mov"])

if uploaded_file is not None:
    # ファイル名が変わったら解析結果をリセット
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.df = None
        st.session_state.last_uploaded_file = uploaded_file.name

    # --- 解析フェーズ ---
    if st.session_state.df is None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        status = st.empty()
        progress_bar = st.progress(0.0)
        data_log = []
        total_angle, prev_angle = 0.0, None
        gx, gy = np.nan, np.nan
        
        for f_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            mask_g = cv2.inRange(hsv, LOWER_GREEN[0], LOWER_GREEN[1])
            con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if con_g:
                c = max(con_g, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0: gx, gy = M["m10"]/M["m00"], M["m01"]/M["m00"]

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

            data_log.append({"t": f_idx/fps, "x": total_angle*(radius_cm/100.0), "gx": gx, "gy": gy, "bx": bx, "by": by})
            if f_idx % 20 == 0: progress_bar.progress(min((f_idx/total_frames), 1.0))

        df_calc = pd.DataFrame(data_log).interpolate().ffill().bfill().fillna(0.0)
        if len(df_calc) > 10:
            df_calc["x"] = savgol_filter(df_calc["x"], 11, 2)
            df_calc["v"] = savgol_filter(df_calc["x"].diff().fillna(0)*fps, 21, 2)
            df_calc["a"] = savgol_filter(df_calc["v"].diff().fillna(0)*fps, 21, 2)
            df_calc["F"] = float(mass) * df_calc["a"]
        
        # 解析結果をセッションに保存
        st.session_state.df = df_calc
        st.session_state.fps = fps
        st.session_state.video_size = (w_orig, h_orig)
        st.session_state.video_path = tfile.name
        cap.release()
        status.success("解析完了！データを保持しています。")

    # --- 保存されたデータを使用して表示 ---
    df = st.session_state.df
    t_max = float(df["t"].max())
    x_min, x_max = float(df["x"].min()), float(df["x"].max())
    v_min, v_max = float(df["v"].min()), float(df["v"].max())
    a_min, a_max = float(df["a"].min()), float(df["a"].max())
    F_min, F_max = float(df["F"].min()), float(df["F"].max())

    st.subheader("📊 物理グラフプレビュー")
    ps = 450
    col1, col2 = st.columns(2)
    with col1: st.image(create_graph_image(df, "t", "x", "t", "x", "s", "m", "blue", ps, 0, t_max, x_min, x_max), channels="BGR")
    with col2: st.image(create_graph_image(df, "t", "v", "t", "v", "s", "m/s", "red", ps, 0, t_max, v_min, v_max), channels="BGR")
    with col1: st.image(create_graph_image(df, "t", "a", "t", "a", "s", "m/s^2", "green", ps, 0, t_max, a_min, a_max), channels="BGR")
    with col2: st.image(create_graph_image(df, "x", "F", "x", "F", "m", "N", "purple", ps, x_min, x_max, F_min, F_max), channels="BGR")

    # --- 仕事Wの算出（ここを入力しても再解析されない） ---
    st.divider()
    st.subheader("🔬 エネルギー解析：仕事 $W$")
    c_in1, c_in2, c_res = st.columns([2, 2, 3])
    with c_in1: t_start = st.number_input("開始時刻 [s]", 0.0, t_max, 0.0, 0.01)
    with c_in2: t_end = st.number_input("終了時刻 [s]", 0.0, t_max, t_max, 0.01)
    
    df_w = df[(df['t'] >= t_start) & (df['t'] <= t_end)].sort_values('t')
    if len(df_w) > 1:
        work_val = np.trapz(df_w['F'].values, df_w['x'].values)
        with c_res:
            st.metric(label="仕事 $W$ (Work)", value=f"{float(work_val):.4f} J")
            st.caption(f"変位 $\Delta x$: {float(df_w['x'].iloc[-1] - df_w['x'].iloc[0]):.3f} m")

    # --- データ保存 ---
    st.divider()
    st.download_button("📊 CSVデータを保存", df[["t","x","v","a","F"]].to_csv(index=False).encode('utf_8_sig'), "kinema_cart_data.csv")
