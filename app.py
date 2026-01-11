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

# --- グラフ描画関数 ---
def create_dynamic_graph(df_sub, x_col, y_col, xlabel, ylabel, color, size, x_max, y_min, y_max):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    if len(df_sub) > 0:
        ax.plot(df_sub[x_col], df_sub[y_col], color=color, linewidth=2)
        ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=40)
    
    ax.set_title(f"{ylabel}-{xlabel}", fontsize=14, fontweight='bold')
    ax.set_xlim(0, x_max if x_max > 0 else 1)
    ax.set_ylim(y_min, y_max if y_max > y_min else y_min + 1)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor='white')
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    return cv2.resize(img, (size, size))

# --- アプリ設定 ---
st.set_page_config(page_title="Physics Lab: Cart Analysis", layout="wide")
st.title("🚀 物理実験：台車の運動解析システム")

st.sidebar.header("実験パラメータ")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mass = 0.1 # kg
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
    progress_bar = st.progress(0)
    
    # --- Step 1: 解析 ---
    status.info("Step 1: 映像解析中...")
    data_log = []
    total_angle = 0.0
    prev_angle = None
    gx, gy = np.nan, np.nan
    frame_count = 0
    frames_tracked = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 緑中心追跡
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

            # 描画
            cv2.circle(frame, (int(gx), int(gy)), mask_size, (255, 255, 255), 2)
            cv2.circle(frame, (int(gx), int(gy)), 5, (0, 255, 0), -1)
            if pd.notna(bx):
                cv2.circle(frame, (int(bx), int(by)), 5, (255, 0, 255), -1)
                cv2.line(frame, (int(gx), int(gy)), (int(bx), int(by)), (255, 255, 255), 1)

        if pd.notna(gx) and pd.notna(bx):
            current_angle = np.arctan2(by - gy, bx - gx)
            if prev_angle is not None:
                diff = current_angle - prev_angle
                if diff > np.pi: diff -= 2 * np.pi
                if diff < -np.pi: diff += 2 * np.pi
                # 右回転（CW）を正にするため、OpenCV座標系ではマイナス
                total_angle -= diff 
            prev_angle = current_angle

        frames_tracked.append(frame)
        data_log.append({"t": frame_count/fps, "x": total_angle * (radius_cm/100)})
        frame_count += 1
        if frame_count % 10 == 0: progress_bar.progress(min(frame_count / total_frames * 0.3, 0.3))
    cap.release()

    # --- Step 2: 物理量計算 ---
    status.info("Step 2: 物理量計算中...")
    df = pd.DataFrame(data_log).interpolate().ffill().bfill()
    # 平滑化処理
    win_v, win_a = 15, 31
    df["x"] = savgol_filter(df["x"], win_v, 2)
    df["v"] = savgol_filter(df["x"].diff().fillna(0) * fps, win_a, 2)
    df["a"] = savgol_filter(df["v"].diff().fillna(0) * fps, win_a, 2)
    df["F"] = mass * df["a"]

    # 画面にグラフを表示（動画生成を待つ間用）
    st.subheader("📊 解析結果プレビュー")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.write("x-t / v-t")
        st.line_chart(df.set_index("t")[["x", "v"]])
    with col_g2:
        st.write("a-t / F-x")
        st.line_chart(df.set_index("t")["a"])
        st.line_chart(df.set_index("x")["F"])

    # --- Step 3: 動画合成 ---
    status.info("Step 3: グラフ動画を合成中 (時間がかかります)...")
    graph_size = w_orig // 4
    header_h = graph_size + 60 # グラフ + 数値表示エリア
    new_h = h_orig + header_h
    
    final_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    out = cv2.VideoWriter(final_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_orig, new_h))

    # スケール固定用
    x_max, t_max = df["x"].max(), df["t"].max()
    v_min, v_max = df["v"].min(), df["v"].max()
    a_min, a_max = df["a"].min(), df["a"].max()
    F_min, F_max = df["F"].min(), df["F"].max()

    for i in range(len(df)):
        canvas = np.zeros((new_h, w_orig, 3), dtype=np.uint8)
        df_s = df.iloc[:i+1]
        
        # 4枚のグラフ作成
        g1 = create_dynamic_graph(df_s, "t", "x", "t", "x", "blue", graph_size, t_max, 0, x_max)
        g2 = create_dynamic_graph(df_s, "t", "v", "t", "v", "red", graph_size, t_max, v_min, v_max)
        g3 = create_dynamic_graph(df_s, "t", "a", "t", "a", "green", graph_size, t_max, a_min, a_max)
        g4 = create_dynamic_graph(df_s, "x", "F", "x", "F", "purple", graph_size, x_max, F_min, F_max)

        # グラフ配置
        canvas[0:graph_size, 0:graph_size] = g1
        canvas[0:graph_size, graph_size:graph_size*2] = g2
        canvas[0:graph_size, graph_size*2:graph_size*3] = g3
        canvas[0:graph_size, graph_size*3:graph_size*4] = g4

        # 数値表示 (グラフの直下)
        font = cv2.FONT_HERSHEY_SIMPLEX
        row_y = graph_size + 40
        curr = df.iloc[i]
        cv2.putText(canvas, f"x:{curr['x']:.3f}m", (10, row_y), font, 0.7, (255,255,255), 2)
        cv2.putText(canvas, f"v:{curr['v']:.2f}m/s", (graph_size+10, row_y), font, 0.7, (255,255,255), 2)
        cv2.putText(canvas, f"a:{curr['a']:.2f}m/s2", (graph_size*2+10, row_y), font, 0.7, (255,255,255), 2)
        cv2.putText(canvas, f"F:{curr['F']:.3f}N", (graph_size*3+10, row_y), font, 0.7, (255,255,255), 2)

        # 元動画
        canvas[header_h:new_h, 0:w_orig] = frames_tracked[i]
        
        # 時刻t表示 (右下)
        cv2.putText(canvas, f"t: {curr['t']:.2f} s", (w_orig-150, new_h-30), font, 1.0, (255,255,255), 2)

        out.write(canvas)
        if i % 10 == 0: progress_bar.progress(0.3 + (i / len(df)) * 0.7)

    out.release()
    status.success("全ての解析が完了しました！")

    # ダウンロードセクション
    st.divider()
    csv = df[["t", "x", "v", "a", "F"]].to_csv(index=False).encode('utf_8_sig')
    st.download_button("📊 CSVデータを保存", csv, "physics_data.csv", "text/csv")
    with open(final_video_path, "rb") as v:
        st.download_button("🎥 物理解析動画を保存", v, "physics_analysis.mp4", "video/mp4")

    os.remove(tfile.name)
