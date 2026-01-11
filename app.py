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

# --- グラフ描画関数 (X軸・Y軸ともに動的な範囲に対応) ---
def create_graph_image(df_sub, x_col, y_col, x_label_text, y_label_text, x_unit, y_unit, color, size, x_min, x_max, y_min, y_max):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    
    if len(df_sub) > 0:
        ax.plot(df_sub[x_col], df_sub[y_col], color=color, linewidth=2)
        ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=50)
    
    ax.set_title(f"${y_label_text}$ - ${x_label_text}$", fontsize=16, fontweight='bold')
    ax.set_xlabel(f"${x_label_text}$ [{x_unit}]", fontsize=14)
    ax.set_ylabel(f"${y_label_text}$ [{y_unit}]", fontsize=14)
    
    # X軸の範囲設定
    x_range = x_max - x_min
    if x_range <= 0: x_range = 1
    ax.set_xlim(x_min - x_range*0.05, x_max + x_range*0.05)
    
    # Y軸の範囲設定
    y_range = y_max - y_min
    if y_range <= 0: y_range = 1
    ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    # 原点を通る軸（0の線）を強調して変位を分かりやすく
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    if x_col != 't': # x軸が変位の場合
        ax.axvline(0, color='black', linewidth=1, alpha=0.5)
        
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor='white')
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    return cv2.resize(img, (size, size))

st.set_page_config(page_title="Physics Lab Pro", layout="wide")
st.title("🚀 物理実験：台車の運動解析システム (変位対応版)")

# --- サイドバー設定 ---
st.sidebar.header("実験パラメータ")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mass = 0.1 
mask_size = st.sidebar.slider("解析エリア半径 (px)", 50, 400, 200, 10)

LOWER_GREEN = (np.array([35, 50, 50]), np.array([85, 255, 255]))
LOWER_PINK = (np.array([140, 40, 40]), np.array([180, 255, 255]))

uploaded_file = st.file_uploader("実験動画を選択 (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    status = st.empty()
    progress_bar = st.progress(0.0)
    
    status.info("Step 1: 映像解析中...")
    data_log = []
    total_angle = 0.0
    prev_angle = None
    gx, gy = np.nan, np.nan
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_g = cv2.inRange(hsv, LOWER_GREEN[0], LOWER_GREEN[1])
        con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if con_g:
            c = max(con_g, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                curr_x, curr_y = M["m10"]/M["m00"], M["m01"]/M["m00"]
                if pd.isna(gx) or np.hypot(curr_x - gx, curr_y - gy) < (w_orig/3):
                    gx, gy = curr_x, curr_y

        bx, by = np.nan, np.nan
        if pd.notna(gx):
            circle_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            cv2.circle(circle_mask, (int(gx), int(gy)), mask_size, 255, -1)
            mask_p = cv2.inRange(cv2.bitwise_and(hsv, hsv, mask=circle_mask), LOWER_PINK[0], LOWER_PINK[1])
            con_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if con_p:
                cp = max(con_p, key=cv2.contourArea)
                Mp = cv2.moments(cp)
                if Mp["m00"] != 0: bx, by = Mp["m10"]/Mp["m00"], Mp["m01"]/Mp["m00"]

        if pd.notna(gx) and pd.notna(bx):
            current_angle = np.arctan2(by - gy, bx - gx)
            if prev_angle is not None:
                diff = current_angle - prev_angle
                if diff > np.pi: diff -= 2 * np.pi
                if diff < -np.pi: diff += 2 * np.pi
                total_angle += diff 
            prev_angle = current_angle

        data_log.append({"t": frame_count/fps, "x": total_angle * (radius_cm/100), "gx": gx, "gy": gy, "bx": bx, "by": by})
        frame_count += 1
        if frame_count % 10 == 0: progress_bar.progress(min(frame_count / total_frames * 0.3, 0.3))
    cap.release()

    status.info("Step 2: 物理量計算中...")
    df = pd.DataFrame(data_log).interpolate().ffill().bfill()
    df["x"] = savgol_filter(df["x"], 15, 2)
    df["v"] = savgol_filter(df["x"].diff().fillna(0) * fps, 31, 2)
    df["a"] = savgol_filter(df["v"].diff().fillna(0) * fps, 31, 2)
    df["F"] = mass * df["a"]

    # 全データの最小・最大を取得して軸を固定
    t_min, t_max = 0, df["t"].max()
    x_min, x_max = df["x"].min(), df["x"].max()
    v_min, v_max = df["v"].min(), df["v"].max()
    a_min, a_max = df["a"].min(), df["a"].max()
    F_min, F_max = df["F"].min(), df["F"].max()

    st.subheader("📊 物理グラフプレビュー")
    plot_size = 500
    st.image(create_graph_image(df, "t", "x", "t", "x", "s", "m", "blue", plot_size, t_min, t_max, x_min, x_max), channels="BGR")
    st.image(create_graph_image(df, "t", "v", "t", "v", "s", "m/s", "red", plot_size, t_min, t_max, v_min, v_max), channels="BGR")
    st.image(create_graph_image(df, "t", "a", "t", "a", "s", "m/s^2", "green", plot_size, t_min, t_max, a_min, a_max), channels="BGR")
    st.image(create_graph_image(df, "x", "F", "x", "F", "m", "N", "purple", plot_size, x_min, x_max, F_min, F_max), channels="BGR")

    # --- Step 3: 動画合成 ---
    status.info("Step 3: 動画を合成中...")
    graph_v_size = w_orig // 4
    header_h = graph_v_size + 100 
    new_h = h_orig + header_h
    
    final_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    out = cv2.VideoWriter(final_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_orig, new_h))

    cap_retry = cv2.VideoCapture(tfile.name)
    font_italic = cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC

    for i in range(len(df)):
        ret, frame = cap_retry.read()
        if not ret: break
        
        canvas = np.zeros((new_h, w_orig, 3), dtype=np.uint8)
        df_s = df.iloc[:i+1]
        curr = df.iloc[i]

        # 4枚のグラフ
        g1 = create_graph_image(df_s, "t", "x", "t", "x", "s", "m", "blue", graph_v_size, t_min, t_max, x_min, x_max)
        g2 = create_graph_image(df_s, "t", "v", "t", "v", "s", "m/s", "red", graph_v_size, t_min, t_max, v_min, v_max)
        g3 = create_graph_image(df_s, "t", "a", "t", "a", "s", "m/s^2", "green", graph_v_size, t_min, t_max, a_min, a_max)
        g4 = create_graph_image(df_s, "x", "F", "x", "F", "m", "N", "purple", graph_v_size, x_min, x_max, F_min, F_max)

        canvas[0:graph_v_size, 0:graph_v_size] = g1
        canvas[0:graph_v_size, graph_v_size:graph_v_size*2] = g2
        canvas[0:graph_v_size, graph_v_size*2:graph_v_size*3] = g3
        canvas[0:graph_v_size, graph_v_size*3:graph_v_size*4] = g4

        # 数値表示
        y_text = graph_v_size + 60
        labels = [
            f"x: {curr['x']:7.3f} m",
            f"v: {curr['v']:7.2f} m/s",
            f"a: {curr['a']:7.2f} m/s2",
            f"F: {curr['F']:7.3f} N"
        ]
        for idx, text in enumerate(labels):
            t_size = cv2.getTextSize(text, font_italic, 1.0, 2)[0]
            start_x = idx * graph_v_size + (graph_v_size - t_size[0]) // 2
            cv2.putText(canvas, text, (start_x, y_text), font_italic, 1.0, (255, 255, 255), 2)

        # トラッキング描画
        if pd.notna(curr['gx']):
            cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), mask_size, (255, 255, 255), 2)
            cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), 6, (0, 255, 0), -1)
            if pd.notna(curr['bx']):
                cv2.circle(frame, (int(curr['bx']), int(curr['by'])), 6, (255, 0, 255), -1)
                cv2.line(frame, (int(curr['gx']), int(curr['gy'])), (int(curr['bx']), int(curr['by'])), (255, 255, 255), 1)

        canvas[header_h:new_h, 0:w_orig] = frame
        t_text = f"t: {curr['t']:6.2f} s"
        t_size = cv2.getTextSize(t_text, font_italic, 1.2, 3)[0]
        cv2.putText(canvas, t_text, (w_orig - t_size[0] - 20, new_h - 40), font_italic, 1.2, (255, 255, 255), 3)

        out.write(canvas)
        if i % 10 == 0: progress_bar.progress(0.3 + (i / len(df)) * 0.7)

    cap_retry.release()
    out.release()
    status.success("解析完了！")

    st.divider()
    csv_data = df[["t", "x", "v", "a", "F"]].to_csv(index=False).encode('utf_8_sig')
    st.download_button(label="📊 CSVデータを保存", data=csv_data, file_name="physics_data_displacement.csv", mime="text/csv")
    with open(final_video_path, "rb") as v:
        st.download_button(label="🎥 解析済み動画を保存", data=v, file_name="physics_analysis_displacement.mp4", mime="video/mp4")

    os.remove(tfile.name)
