import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tempfile
import os

st.set_page_config(page_title="台車解析アプリ V1 (Pink)", layout="wide")
st.title("🏃‍♂️ 物理実験：台車の速度解析 (ピンクマーカー版)")

st.sidebar.header("設定")
radius = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1) # 3.2cmの半分=1.6

# --- 色の定義 (緑とピンク) ---
LOWER_GREEN = (np.array([30, 40, 40]), np.array([100, 255, 255]))
LOWER_PINK = (np.array([140, 50, 50]), np.array([175, 255, 255]))

uploaded_file = st.file_uploader("iPadで撮った動画を選択してください", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w_orig, h_orig))

    st.info("動画を解析・生成中です...")
    progress_bar = st.progress(0)
    
    data_log = []
    total_angle = 0.0
    prev_angle = None
    last_gx, last_gy = np.nan, np.nan
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_roi = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.rectangle(mask_roi, (0, h_orig // 2), (w_orig, h_orig), 255, -1)
        hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask_roi)
        
        gx = gy = bx = by = np.nan
        
        # 緑（中心）
        mask_g = cv2.inRange(hsv_masked, LOWER_GREEN[0], LOWER_GREEN[1])
        con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if con_g:
            M = cv2.moments(max(con_g, key=cv2.contourArea))
            if M["m00"] != 0:
                gx, gy = M["m10"]/M["m00"], M["m01"]/M["m00"]
                last_gx, last_gy = gx, gy
                cv2.circle(frame, (int(gx), int(gy)), 10, (0, 255, 0), -1)
        else: gx, gy = last_gx, last_gy

        # ピンク（円周点）
        mask_p = cv2.inRange(hsv_masked, LOWER_PINK[0], LOWER_PINK[1])
        con_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if con_p:
            M = cv2.moments(max(con_p, key=cv2.contourArea))
            if M["m00"] != 0:
                bx, by = M["m10"]/M["m00"], M["m01"]/M["m00"]
                # ピンク色で描画
                cv2.circle(frame, (int(bx), int(by)), 10, (147, 20, 255), -1)

        if pd.notna(gx) and pd.notna(bx):
            current_angle = np.arctan2(by - gy, bx - gx)
            if prev_angle is not None:
                diff = current_angle - prev_angle
                if diff > np.pi: diff -= 2 * np.pi
                if diff < -np.pi: diff += 2 * np.pi
                total_angle += diff
            prev_angle = current_angle

        out_writer.write(frame)
        data_log.append({"Time": frame_count/fps, "Distance": abs(total_angle) * radius})
        frame_count += 1
        if frame_count % 10 == 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
    cap.release()
    out_writer.release()
    
    df = pd.DataFrame(data_log).interpolate().fillna(method='bfill')
    if len(df) > 31:
        df["Distance"] = savgol_filter(df["Distance"], window_length=15, polyorder=2)
        raw_speed = df["Distance"].diff().fillna(0) * fps
        df["Speed"] = savgol_filter(raw_speed, window_length=31, polyorder=2)
    else:
        df["Speed"] = df["Distance"].diff().fillna(0) * fps
    df["Speed"] = df["Speed"].clip(lower=0)

    st.success("解析完了！")
    
    col_metrics, col_charts = st.columns([1, 2])
    with col_metrics:
        st.subheader("📊 最終計測値")
        st.metric("合計時間", f"{df['Time'].iloc[-1]:.2f} s")
        st.metric("走行距離", f"{df['Distance'].iloc[-1]:.1f} cm")
        st.metric("最大速度", f"{df['Speed'].max():.1f} cm/s")
        st.metric("平均速度", f"{(df['Distance'].iloc[-1]/df['Time'].iloc[-1]) if df['Time'].iloc[-1]>0 else 0:.1f} cm/s")

    with col_charts:
        st.subheader("📈 解析グラフ")
        tab1, tab2 = st.tabs(["速度 (Speed)", "距離 (Distance)"])
        with tab1: st.line_chart(df.set_index("Time")["Speed"])
        with tab2: st.line_chart(df.set_index("Time")["Distance"])

    st.divider()
    st.subheader("📁 データのダウンロード")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv = df.to_csv(index=False).encode('utf_8_sig')
        st.download_button("📊 CSVデータを保存", data=csv, file_name="result_data.csv", mime="text/csv")
    with dl_col2:
        with open(out_video_path, "rb") as v_file:
            st.download_button("🎥 解析済み動画を保存", data=v_file, file_name="analyzed_video.mp4", mime="video/mp4")

    os.remove(tfile.name)
