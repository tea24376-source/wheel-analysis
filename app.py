import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tempfile
import os

# アプリのタイトル
st.set_page_config(page_title="台車解析アプリ V1 (動画保存版)", layout="wide")
st.title("🏃‍♂️ 物理実験：台車の速度解析 (動画保存対応版)")

st.sidebar.header("設定")
radius = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.5, 0.1)

# 色の設定 (V1: 緑と青)
LOWER_GREEN = (np.array([30, 40, 40]), np.array([100, 255, 255]))
LOWER_BLUE = (np.array([90, 50, 50]), np.array([150, 255, 255]))

uploaded_file = st.file_uploader("iPadで撮った動画を選択してください", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None: fps = 30
    
    # 元動画のサイズを取得
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- 動画書き出し用の準備 ---
    # 保存用の一時ファイルを作成
    out_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    # OpenCVのVideoWriter（iPadで再生しやすい mp4v コーデック）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w_orig, h_orig))

    st.info(f"動画を解析・生成中です。完了までしばらくお待ちください...")
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
        
        # 指対策（上半分をマスク）
        mask_roi = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.rectangle(mask_roi, (0, h_orig // 2), (w_orig, h_orig), 255, -1)
        hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask_roi)
        
        gx = gy = bx = by = np.nan
        
        # 緑（中心）の検出
        mask_g = cv2.inRange(hsv_masked, LOWER_GREEN[0], LOWER_GREEN[1])
        con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if con_g:
            M = cv2.moments(max(con_g, key=cv2.contourArea))
            if M["m00"] != 0:
                gx, gy = M["m10"]/M["m00"], M["m01"]/M["m00"]
                last_gx, last_gy = gx, gy
                # 動画に中心点を描画 (緑色の丸)
                cv2.circle(frame, (int(gx), int(gy)), 10, (0, 255, 0), -1)
        else: gx, gy = last_gx, last_gy

        # 青（円周点）の検出
        mask_b = cv2.inRange(hsv_masked, LOWER_BLUE[0], LOWER_BLUE[1])
        con_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if con_b:
            M = cv2.moments(max(con_b, key=cv2.contourArea))
            if M["m00"] != 0:
                bx, by = M["m10"]/M["m00"], M["m01"]/M["m00"]
                # 動画に円周点を描画 (青色の丸)
                cv2.circle(frame, (int(bx), int(by)), 10, (255, 0, 0), -1)

        # 角度計算
        if pd.notna(gx) and pd.notna(bx):
            current_angle = np.arctan2(by - gy, bx - gx)
            if prev_angle is not None:
                diff = current_angle - prev_angle
                if diff > np.pi: diff -= 2 * np.pi
                if diff < -np.pi: diff += 2 * np.pi
                total_angle += diff
            prev_angle = current_angle

        # 加工したフレームを保存用動画に書き込む
        out_writer.write(frame)

        data_log.append({"Time": frame_count/fps, "Distance": abs(total_angle) * radius})
        frame_count += 1
        if frame_count % 10 == 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
    cap.release()
    out_writer.release() # ここで動画ファイルが完成
    
    # データ処理（平滑化）
    df = pd.DataFrame(data_log).interpolate().fillna(method='bfill')
    if len(df) > 31:
        df["Distance"] = savgol_filter(df["Distance"], window_length=15, polyorder=2)
        raw_speed = df["Distance"].diff().fillna(0) * fps
        df["Speed"] = savgol_filter(raw_speed, window_length=31, polyorder=2)
    else:
        df["Speed"] = df["Distance"].diff().fillna(0) * fps
    df["Speed"] = df["Speed"].clip(lower=0)

    # --- 結果表示画面 ---
    st.success("解析が完了しました！")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("速度の推移 (cm/s)")
        st.line_chart(df.set_index("Time")["Speed"])
    with col2:
        st.subheader("走行距離の推移 (cm)")
        st.line_chart(df.set_index("Time")["Distance"])
    
    st.divider()
    st.subheader("📁 データの保存")
    dl_col1, dl_col2 = st.columns(2)
    
    with dl_col1:
        # CSVダウンロード
        csv = df.to_csv(index=False).encode('utf_8_sig')
        st.download_button("📊 CSVデータを保存", data=csv, file_name="result_data.csv", mime="text/csv")
    
    with dl_col2:
        # 解析済み動画のダウンロード
        with open(out_video_path, "rb") as v_file:
            st.download_button("🎥 解析済み動画を保存", data=v_file, file_name="analyzed_video.mp4", mime="video/mp4")

    # 一時ファイルの削除（サーバーの容量確保）
    os.remove(tfile.name)
