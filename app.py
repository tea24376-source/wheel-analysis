import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tempfile
import os
import matplotlib.pyplot as plt
import io

# Matplotlibのバックエンド設定
plt.switch_backend('Agg')

# --- グラフ画像を生成する関数 (動的描画用) ---
def create_dynamic_graph(df_sub, x_col, y_col, title, color, target_width, target_height, x_max, y_max):
    # 常に同じスケールのグラフ枠を作る
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    
    if len(df_sub) > 0:
        ax.plot(df_sub[x_col], df_sub[y_col], color=color, linewidth=2)
        # 現在地点に点を打つ
        ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=30)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max * 1.1) # 少し余裕を持たせる
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 画像に変換
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=False, facecolor='white')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    plt.close(fig)
    
    return cv2.resize(img, (target_width, target_height))

# --- メインアプリ ---
st.set_page_config(page_title="台車解析アプリ Final Pro", layout="wide")
st.title("🏃‍♂️ 物理実験：台車の速度解析 (グラフ動的合成版)")

st.sidebar.header("設定")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mask_size = st.sidebar.slider("解析エリアの半径 (px)", 50, 400, 200, 10)

# 色の設定
LOWER_GREEN = (np.array([35, 50, 50]), np.array([85, 255, 255]))
LOWER_PINK = (np.array([140, 40, 40]), np.array([180, 255, 255]))

uploaded_file = st.file_uploader("動画を選択してください", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # --- Step 1: 解析 ---
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 中間保存用
    temp_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (w_orig, h_orig))

    status = st.empty()
    progress_bar = st.progress(0)
    
    data_log = []
    total_angle = 0.0
    prev_angle = None
    gx, gy = np.nan, np.nan
    frame_count = 0

    status.info("Step 1/3: 台車の動きを追跡中...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_g = cv2.inRange(hsv, LOWER_GREEN[0], LOWER_GREEN[1])
        con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_gx, best_gy = np.nan, np.nan
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
            hsv_m = cv2.bitwise_and(hsv, hsv, mask=circle_mask)
            mask_p = cv2.inRange(hsv_m, LOWER_PINK[0], LOWER_PINK[1])
            con_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if con_p:
                c_p = max(con_p, key=cv2.contourArea)
                M_p = cv2.moments(c_p)
                if M_p["m00"] != 0: bx, by = M_p["m10"]/M_p["m00"], M_p["m01"]/M_p["m00"]

            # 動画へのガイド描画
            cv2.circle(frame, (int(gx), int(gy)), mask_size, (255, 255, 255), 2)
            cv2.circle(frame, (int(gx), int(gy)), 8, (0, 255, 0), -1)
            if pd.notna(bx):
                cv2.circle(frame, (int(bx), int(by)), 8, (147, 20, 255), -1)
                cv2.line(frame, (int(gx), int(gy)), (int(bx), int(by)), (255, 255, 255), 2)

        if pd.notna(gx) and pd.notna(bx):
            current_angle = np.arctan2(by - gy, bx - gx)
            if prev_angle is not None:
                diff = current_angle - prev_angle
                if diff > np.pi: diff -= 2 * np.pi
                if diff < -np.pi: diff += 2 * np.pi
                total_angle += diff
            prev_angle = current_angle

        temp_writer.write(frame)
        data_log.append({"Time": frame_count/fps, "Distance": abs(total_angle) * radius_cm})
        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames * 0.4, 0.4))
            
    cap.release()
    temp_writer.release()
    
    # --- Step 2: データ平滑化 ---
    status.info("Step 2/3: データを計算中...")
    df = pd.DataFrame(data_log).interpolate().ffill().bfill()
    if len(df) > 31:
        df["Distance"] = savgol_filter(df["Distance"], 15, 2)
        df["Speed"] = savgol_filter(df["Distance"].diff().fillna(0)*fps, 31, 2)
    else:
        df["Speed"] = df["Distance"].diff().fillna(0)*fps
    df["Speed"] = df["Speed"].clip(lower=0)

    # グラフの最大値を固定するために取得
    x_max = df["Time"].max()
    s_max = df["Speed"].max()
    d_max = df["Distance"].max()
    graph_w, graph_h = int(w_orig * 0.35), int(h_orig * 0.28)

    # --- Step 3: 動的グラフの合成 ---
    status.info("Step 3/3: グラフを動画に書き込み中... (少し時間がかかります)")
    final_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    final_writer = cv2.VideoWriter(final_video_path, fourcc, fps, (w_orig, h_orig))
    cap_temp = cv2.VideoCapture(temp_video_path)

    for i in range(len(df)):
        ret, frame = cap_temp.read()
        if not ret: break

        # そのフレームまでのデータを抽出
        df_sub = df.iloc[:i+1]
        
        # グラフ作成
        s_img = create_dynamic_graph(df_sub, "Time", "Speed", "Speed (cm/s)", "red", graph_w, graph_h, x_max, s_max)
        d_img = create_dynamic_graph(df_sub, "Time", "Distance", "Distance (cm)", "blue", graph_w, graph_h, x_max, d_max)

        # 合成（右上）
        m = 15
        frame[m:m+graph_h, w_orig-m-graph_w:w_orig-m] = s_img
        frame[m*2+graph_h:m*2+graph_h*2, w_orig-m-graph_w:w_orig-m] = d_img
        
        final_writer.write(frame)
        if i % 10 == 0: progress_bar.progress(min(0.4 + (i / len(df)) * 0.6, 1.0))

    cap_temp.release()
    final_writer.release()
    status.success("すべての解析と動画生成が完了しました！")
    
    # UI表示とダウンロード
    st.metric("走行距離", f"{df['Distance'].iloc[-1]:.1f} cm")
    with open(final_video_path, "rb") as v:
        st.download_button("🎥 グラフが動く動画を保存", data=v, file_name="moving_graph_analysis.mp4", mime="video/mp4")
    
    os.remove(tfile.name)
    os.remove(temp_video_path)
