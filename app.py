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

# --- グラフ描画関数 (ガード付き) ---
def create_graph_image(df_sub, x_col, y_col, x_label_text, y_label_text, x_unit, y_unit, color, size, x_min, x_max, y_min, y_max):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    
    if len(df_sub) > 0:
        ax.plot(df_sub[x_col], df_sub[y_col], color=color, linewidth=2)
        ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=50)
    
    ax.set_title(f"${y_label_text}$ - ${x_label_text}$", fontsize=16, fontweight='bold')
    ax.set_xlabel(f"${x_label_text}$ [{x_unit}]", fontsize=14)
    ax.set_ylabel(f"${y_label_text}$ [{y_unit}]", fontsize=14)
    
    # 軸範囲のエラー防止
    x_range = max(float(x_max - x_min), 0.001)
    ax.set_xlim(x_min - x_range*0.05, x_max + x_range*0.05)
    
    y_range = max(float(y_max - y_min), 0.001)
    ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    if x_col != 't':
        ax.axvline(0, color='black', linewidth=1, alpha=0.5)
        
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor='white')
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    return cv2.resize(img, (size, size))

st.set_page_config(page_title="CartGrapher Studio", layout="wide")
st.title("🚀 CartGrapher Studio")

# --- サイドバー：Kinema-Cart設定 ---
st.sidebar.header("Kinema-Cart 設定")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mass = st.sidebar.number_input("台車の質量 $m$ (kg)", value=0.1, min_value=0.001, step=0.01, format="%.3f")
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
    
    # --- Step 1: 解析 ---
    data_log = []
    total_angle = 0.0
    prev_angle = None
    gx, gy = np.nan, np.nan
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 緑（中心）
        mask_g = cv2.inRange(hsv, LOWER_GREEN[0], LOWER_GREEN[1])
        con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if con_g:
            c = max(con_g, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                gx, gy = M["m10"]/M["m00"], M["m01"]/M["m00"]

        # ピンク（外周）
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

    # --- Step 2: 物理量計算 ---
    df = pd.DataFrame(data_log).interpolate().ffill().bfill().fillna(0)
    df["x"] = savgol_filter(df["x"], 15, 2)
    df["v"] = savgol_filter(df["x"].diff().fillna(0) * fps, 31, 2)
    df["a"] = savgol_filter(df["v"].diff().fillna(0) * fps, 31, 2)
    df["F"] = mass * df["a"]
    # 再度NaN埋め
    df = df.fillna(0)

    # 全域のスケール取得
    t_min, t_max = 0, float(df["t"].max())
    x_min, x_max = float(df["x"].min()), float(df["x"].max())
    v_min, v_max = float(df["v"].min()), float(df["v"].max())
    a_min, a_max = float(df["a"].min()), float(df["a"].max())
    F_min, F_max = float(df["F"].min()), float(df["F"].max())

    # --- プレビュー表示 ---
    st.subheader("📊 物理グラフプレビュー")
    p_size = 500
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1: st.image(create_graph_image(df, "t", "x", "t", "x", "s", "m", "blue", p_size, t_min, t_max, x_min, x_max), channels="BGR")
    with row1_c2: st.image(create_graph_image(df, "t", "v", "t", "v", "s", "m/s", "red", p_size, t_min, t_max, v_min, v_max), channels="BGR")
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1: st.image(create_graph_image(df, "t", "a", "t", "a", "s", "m/s^2", "green", p_size, t_min, t_max, a_min, a_max), channels="BGR")
    with row2_c2: st.image(create_graph_image(df, "x", "F", "x", "F", "m", "N", "purple", p_size, x_min, x_max, F_min, F_max), channels="BGR")

    # --- ★ 仕事 W の計算セクション ---
    st.divider()
    st.subheader("🔬 エネルギー解析：仕事 $W$")
    st.write("$F-x$ グラフの面積から、指定区間の仕事 $W$ を算出します。")
    
    calc_c1, calc_c2, calc_c3 = st.columns([2, 2, 3])
    with calc_c1: t_s = st.number_input("開始時刻 $t$ [s]", 0.0, t_max, 0.0, 0.1)
    with calc_c2: t_e = st.number_input("終了時刻 $t$ [s]", 0.0, t_max, t_max, 0.1)
    
    df_w = df[(df['t'] >= t_s) & (df['t'] <= t_e)]
    if len(df_w) > 1:
        # 積分計算
        w_val = np.trapz(df_w['F'], df_w['x'])
        with calc_c3:
            st.metric(label="仕事 $W$ [J]", value=f"{w_val:.4f} J")
            st.info(f"区間変位: $\Delta x = {df_w['x'].iloc[-1] - df_w['x'].iloc[0]:.3f}$ m")
    else:
        st.warning("有効な範囲を選択してください")

    # --- Step 3: 動画合成 ---
    status.info("動画生成中...")
    g_v_size = w_orig // 4
    header_h = g_v_size + 100 
    final_v_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    out = cv2.VideoWriter(final_v_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_orig, h_orig + header_h))
    
    cap_re = cv2.VideoCapture(tfile.name)
    font_it = cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC

    for i in range(len(df)):
        ret, frame = cap_re.read()
        if not ret: break
        canvas = np.zeros((h_orig + header_h, w_orig, 3), dtype=np.uint8)
        curr = df.iloc[i]
        
        # 4つのグラフ
        gs = [
            create_graph_image(df.iloc[:i+1], "t", "x", "t", "x", "s", "m", "blue", g_v_size, t_min, t_max, x_min, x_max),
            create_graph_image(df.iloc[:i+1], "t", "v", "t", "v", "s", "m/s", "red", g_v_size, t_min, t_max, v_min, v_max),
            create_graph_image(df.iloc[:i+1], "t", "a", "t", "a", "s", "m/s^2", "green", g_v_size, t_min, t_max, a_min, a_max),
            create_graph_image(df.iloc[:i+1], "x", "F", "x", "F", "m", "N", "purple", g_v_size, x_min, x_max, F_min, F_max)
        ]
        for idx, g_img in enumerate(gs):
            canvas[0:g_v_size, idx*g_v_size:(idx+1)*g_v_size] = g_img

        # 数値表示
        labels = [f"x: {curr['x']:.3f} m", f"v: {curr['v']:.2f} m/s", f"a: {curr['a']:.2f} m/s2", f"F: {curr['F']:.3f} N"]
        for idx, txt in enumerate(labels):
            ts = cv2.getTextSize(txt, font_it, 0.9, 2)[0]
            cv2.putText(canvas, txt, (idx*g_v_size + (g_v_size-ts[0])//2, g_v_size+60), font_it, 0.9, (255,255,255), 2)

        # トラッキング描画
        if pd.notna(curr['gx']):
            cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), mask_size, (200, 200, 200), 2)
            cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), 6, (0, 255, 0), -1)
            if pd.notna(curr['bx']):
                cv2.circle(frame, (int(curr['bx']), int(curr['by'])), 6, (255, 0, 255), -1)
                cv2.line(frame, (int(curr['gx']), int(curr['gy'])), (int(curr['bx']), int(curr['by'])), (255, 255, 255), 1)

        canvas[header_h:, 0:w_orig] = frame
        t_txt = f"t: {curr['t']:.2f} s"
        t_sz = cv2.getTextSize(t_txt, font_it, 1.1, 2)[0]
        cv2.putText(canvas, t_txt, (w_orig - t_sz[0] - 20, h_orig + header_h - 30), font_it, 1.1, (255, 255, 255), 2)
        out.write(canvas)
    
    cap_re.release()
    out.release()
    status.success("すべての解析が完了しました！")

    st.divider()
    st.download_button("📊 CSVデータを保存", df[["t", "x", "v", "a", "F"]].to_csv(index=False).encode('utf_8_sig'), "kinema_cart_data.csv", "text/csv")
    with open(final_v_path, "rb") as f:
        st.download_button("🎥 解析動画を保存", f, "cart_grapher_output.mp4", "video/mp4")
    os.remove(tfile.name)
