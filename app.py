import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tempfile
import os
import matplotlib.pyplot as plt
import io

# Matplotlibのバックエンドを非対話モードに設定（サーバーエラー防止）
plt.switch_backend('Agg')

# --- グラフ画像を生成するヘルパー関数 ---
def create_graph_overlay(df, x_col, y_col, title, color, target_width, target_height):
    # グラフを描画
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot(df[x_col], df[y_col], color=color, linewidth=2)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # メモリ上の画像バッファに保存
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=False, facecolor='white')
    buf.seek(0)
    # OpenCV形式の画像に変換
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    plt.close(fig) # メモリ解放
    
    # 指定サイズにリサイズ
    img_resized = cv2.resize(img, (target_width, target_height))
    # 枠線をつける
    cv2.rectangle(img_resized, (0,0), (target_width-1, target_height-1), (200,200,200), 2)
    return img_resized

# --- メインアプリ ---
st.set_page_config(page_title="台車解析アプリ Final", layout="wide")
st.title("🏃‍♂️ 物理実験：台車の速度解析 (グラフ動画埋め込み版)")

st.sidebar.header("設定")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mask_size = st.sidebar.slider("解析エリアの半径 (px)", 50, 400, 200, 10)

# 色の設定 (緑とピンク)
LOWER_GREEN = (np.array([35, 50, 50]), np.array([85, 255, 255]))
LOWER_PINK = (np.array([140, 40, 40]), np.array([180, 255, 255]))

uploaded_file = st.file_uploader("iPadで撮った動画を選択してください", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # --- パス1：トラッキング解析 ---
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 中間ファイル（トラッキング描画のみ）
    temp_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (w_orig, h_orig))

    status_text = st.empty()
    status_text.info("Step 1/3: 解析中... (点の追跡)")
    progress_bar = st.progress(0)
    
    data_log = []
    total_angle = 0.0
    prev_angle = None
    gx, gy = np.nan, np.nan
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 緑（中心）追跡
        mask_g = cv2.inRange(hsv, LOWER_GREEN[0], LOWER_GREEN[1])
        con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_gx, best_gy = np.nan, np.nan
        min_dist = float('inf')
        if con_g:
            for c in con_g:
                if cv2.contourArea(c) < 20: continue
                M = cv2.moments(c)
                if M["m00"] != 0:
                    curr_x, curr_y = M["m10"]/M["m00"], M["m01"]/M["m00"]
                    if pd.isna(gx):
                        best_gx, best_gy = curr_x, curr_y
                        break 
                    else:
                        dist = np.hypot(curr_x - gx, curr_y - gy)
                        if dist < min_dist: min_dist, best_gx, best_gy = dist, curr_x, curr_y
            if pd.notna(best_gx):
                if pd.isna(gx) or min_dist < (w_orig / 2): gx, gy = best_gx, best_gy

        # ピンク（円周）検出
        bx, by = np.nan, np.nan
        if pd.notna(gx):
            circle_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            cv2.circle(circle_mask, (int(gx), int(gy)), mask_size, 255, -1)
            hsv_masked = cv2.bitwise_and(hsv, hsv, mask=circle_mask)
            mask_p = cv2.inRange(hsv_masked, LOWER_PINK[0], LOWER_PINK[1])
            con_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if con_p:
                c_p = max(con_p, key=cv2.contourArea)
                M_p = cv2.moments(c_p)
                if M_p["m00"] != 0: bx, by = M_p["m10"]/M_p["m00"], M_p["m01"]/M_p["m00"]

            # ガイド描画
            cv2.circle(frame, (int(gx), int(gy)), mask_size, (255, 255, 255), 2)
            cv2.circle(frame, (int(gx), int(gy)), 8, (0, 255, 0), -1)
            if pd.notna(bx):
                cv2.circle(frame, (int(bx), int(by)), 8, (147, 20, 255), -1)
                cv2.line(frame, (int(gx), int(gy)), (int(bx), int(by)), (255, 255, 255), 2)

        # 角度・データ計算
        if pd.notna(gx) and pd.notna(bx):
            current_angle = np.arctan2(by - gy, bx - gx)
            if prev_angle is not None:
                diff = current_angle - prev_angle
                if diff > np.pi: diff -= 2 * np.pi
                if diff < -np.pi: diff += 2 * np.pi
                total_angle += diff
            prev_angle = current_angle

        temp_writer.write(frame) # 中間ファイルに書き込み
        data_log.append({"Time": frame_count/fps, "Distance": abs(total_angle) * radius_cm})
        frame_count += 1
        if frame_count % 5 == 0: progress_bar.progress(min(frame_count / total_frames * 0.5, 0.5))
            
    cap.release()
    temp_writer.release()
    
    # --- パス2：データ計算とグラフ画像生成 ---
    status_text.info("Step 2/3: データを計算中...")
    df = pd.DataFrame(data_log).interpolate().ffill().bfill()
    if len(df) > 31:
        df["Distance"] = savgol_filter(df["Distance"], window_length=15, polyorder=2)
        raw_speed = df["Distance"].diff().fillna(0) * fps
        df["Speed"] = savgol_filter(raw_speed, window_length=31, polyorder=2)
    else:
        df["Speed"] = df["Distance"].diff().fillna(0) * fps
    df["Speed"] = df["Speed"].clip(lower=0)

    # グラフ画像の生成（動画の横幅の約30%、高さの約25%のサイズで作成）
    graph_w = int(w_orig * 0.3)
    graph_h = int(h_orig * 0.25)
    speed_graph_img = create_graph_overlay(df, "Time", "Speed", "Speed (cm/s)", "red", graph_w, graph_h)
    dist_graph_img = create_graph_overlay(df, "Time", "Distance", "Distance (cm)", "blue", graph_w, graph_h)

    # --- パス3：グラフの合成と最終出力 ---
    status_text.info("Step 3/3: 動画を作成中... (グラフの合成)")
    final_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    final_writer = cv2.VideoWriter(final_video_path, fourcc, fps, (w_orig, h_orig))
    cap_temp = cv2.VideoCapture(temp_video_path)

    frame_idx = 0
    while cap_temp.isOpened():
        ret, frame = cap_temp.read()
        if not ret: break

        # グラフを右上に配置（余白10px）
        margin = 10
        # 速度グラフ（上）
        frame[margin:margin+graph_h, w_orig-margin-graph_w:w_orig-margin] = speed_graph_img
        # 距離グラフ（下）
        frame[margin*2+graph_h:margin*2+graph_h*2, w_orig-margin-graph_w:w_orig-margin] = dist_graph_img
        
        final_writer.write(frame)
        frame_idx += 1
        if frame_idx % 5 == 0: progress_bar.progress(min(0.5 + frame_idx / total_frames * 0.5, 1.0))

    cap_temp.release()
    final_writer.release()
    progress_bar.empty()
    status_text.success("すべての処理が完了しました！")
    
    # --- 結果表示UI ---
    col_metrics, col_charts = st.columns([1, 2])
    with col_metrics:
        st.subheader("📊 計測結果")
        st.metric("走行距離", f"{df['Distance'].iloc[-1]:.1f} cm")
        st.metric("最大速度", f"{df['Speed'].max():.1f} cm/s")
        st.metric("平均速度", f"{(df['Distance'].iloc[-1]/df['Time'].iloc[-1]) if df['Time'].iloc[-1]>0 else 0:.1f} cm/s")
    with col_charts:
        tab1, tab2 = st.tabs(["速度", "距離"])
        with tab1: st.line_chart(df.set_index("Time")["Speed"])
        with tab2: st.line_chart(df.set_index("Time")["Distance"])

    st.divider()
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv = df.to_csv(index=False).encode('utf_8_sig')
        st.download_button("📊 CSV保存", data=csv, file_name="result.csv", mime="text/csv")
    with dl_col2:
        with open(final_video_path, "rb") as v_file:
            st.download_button("🎥 グラフ付き動画を保存", data=v_file, file_name="analyzed_with_graph.mp4", mime="video/mp4")

    # 一時ファイルの削除
    os.remove(tfile.name)
    os.remove(temp_video_path)
