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

def create_graph_image(df_sub, x_col, y_col, x_label_text, y_label_text, x_unit, y_unit, color, size, x_min, x_max, y_min, y_max):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    try:
        if len(df_sub) > 0:
            ax.plot(df_sub[x_col], df_sub[y_col], color=color, linewidth=2)
            ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=50)
        
        ax.set_title(f"${y_label_text}$ - ${x_label_text}$", fontsize=16, fontweight='bold')
        ax.set_xlabel(f"${x_label_text}$ [{x_unit}]", fontsize=14)
        ax.set_ylabel(f"${y_label_text}$ [{y_unit}]", fontsize=14)
        
        # 数値の安全な取得
        x_min, x_max = float(x_min), float(x_max)
        y_min, y_max = float(y_min), float(y_max)
        
        x_range = max(x_max - x_min, 0.001)
        ax.set_xlim(x_min - x_range*0.05, x_max + x_range*0.05)
        
        y_range = max(y_max - y_min, 0.001)
        ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
        
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
    except:
        plt.close(fig)
        return np.zeros((size, size, 3), dtype=np.uint8)

st.set_page_config(page_title="CartGrapher Studio", layout="wide")
st.title("🚀 CartGrapher Studio")

# --- サイドバー ---
st.sidebar.header("Kinema-Cart 設定")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mass = st.sidebar.number_input("台車の質量 $m$ (kg)", value=0.100, min_value=0.001, step=0.001, format="%.3f")
mask_size = st.sidebar.slider("解析エリア半径 (px)", 50, 400, 200, 10)

LOWER_GREEN = (np.array([35, 50, 50]), np.array([85, 255, 255]))
LOWER_PINK = (np.array([140, 40, 40]), np.array([180, 255, 255]))

uploaded_file = st.file_uploader("実験動画を選択 (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    status = st.empty()
    progress_bar = st.progress(0.0)
    
    data_log = []
    total_angle = 0.0
    prev_angle = None
    gx, gy = np.nan, np.nan
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 緑追跡
        mask_g = cv2.inRange(hsv, LOWER_GREEN[0], LOWER_GREEN[1])
        con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if con_g:
            c = max(con_g, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0: gx, gy = M["m10"]/M["m00"], M["m01"]/M["m00"]

        # ピンク追跡
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

        data_log.append({"t": frame_count/fps, "x": total_angle * (radius_cm/100.0), "gx": gx, "gy": gy, "bx": bx, "by": by})
        frame_count += 1
        if frame_count % 15 == 0: progress_bar.progress(min(frame_count / total_frames * 0.3, 0.3))
    cap.release()

    # --- 物理量計算 ---
    df = pd.DataFrame(data_log).interpolate().ffill().bfill().fillna(0)
    if len(df) > 15:
        df["x"] = savgol_filter(df["x"], 15, 2)
        df["v"] = savgol_filter(df["x"].diff().fillna(0) * fps, 31, 2)
        df["a"] = savgol_filter(df["v"].diff().fillna(0) * fps, 31, 2)
        df["F"] = float(mass) * df["a"]
    df = df.fillna(0)

    # 安全な最大・最小取得
    t_max = float(df["t"].max()) if not df.empty else 1.0
    x_min, x_max = float(df["x"].min()), float(df["x"].max())
    v_min, v_max = float(df["v"].min()), float(df["v"].max())
    a_min, a_max = float(df["a"].min()), float(df["a"].max())
    F_min, F_max = float(df["F"].min()), float(df["F"].max())

    # --- プレビュー表示 ---
    st.subheader("📊 物理グラフプレビュー")
    ps = 500
    c1, c2 = st.columns(2)
    with c1: st.image(create_graph_image(df, "t", "x", "t", "x", "s", "m", "blue", ps, 0, t_max, x_min, x_max), channels="BGR")
    with c2: st.image(create_graph_image(df, "t", "v", "t", "v", "s", "m/s", "red", ps, 0, t_max, v_min, v_max), channels="BGR")
    with c1: st.image(create_graph_image(df, "t", "a", "t", "a", "s", "m/s^2", "green", ps, 0, t_max, a_min, a_max), channels="BGR")
    with c2: st.image(create_graph_image(df, "x", "F", "x", "F", "m", "N", "purple", ps, x_min, x_max, F_min, F_max), channels="BGR")

    # --- 仕事Wの算出セクション ---
    st.divider()
    st.subheader("🔬 エネルギー解析：仕事 $W$")
    ec1, ec2, ec3 = st.columns([2, 2, 3])
    with ec1: t_s = st.number_input("開始 $t$ [s]", 0.0, t_max, 0.0, 0.1)
    with ec2: t_e = st.number_input("終了 $t$ [s]", 0.0, t_max, t_max, 0.1)
    
    # 積分計算の実行
    mask = (df['t'] >= t_s) & (df['t'] <= t_e)
    df_w = df[mask].sort_values('t')
    
    if len(df_w) > 1:
        # 仕事 W = ∫ F dx
        work_val = np.trapz(df_w['F'], df_w['x'])
        with ec3:
            st.metric(label="仕事 $W$ [J]", value=f"{float(work_val):.4f} J")
            st.info(f"変位 $\Delta x$: {float(df_w['x'].iloc[-1] - df_w['x'].iloc[0]):.3f} m")
    else:
        st.warning("解析可能な時間範囲を指定してください。")

    # --- 動画合成 ---
    status.info("解析動画を生成中...")
    gv = w_orig // 4
    hh = gv + 100
    final_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    out = cv2.VideoWriter(final_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_orig, h_orig + hh))
    
    cap_re = cv2.VideoCapture(tfile.name)
    font_it = cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC

    for i in range(len(df)):
        ret, frame = cap_re.read()
        if not ret: break
        canvas = np.zeros((h_orig + hh, w_orig, 3), dtype=np.uint8)
        c = df.iloc[i]
        
        # グラフ描画
        imgs = [
            create_graph_image(df.iloc[:i+1], "t", "x", "t", "x", "s", "m", "blue", gv, 0, t_max, x_min, x_max),
            create_graph_image(df.iloc[:i+1], "t", "v", "t", "v", "s", "m/s", "red", gv, 0, t_max, v_min, v_max),
            create_graph_image(df.iloc[:i+1], "t", "a", "t", "a", "s", "m/s^2", "green", gv, 0, t_max, a_min, a_max),
            create_graph_image(df.iloc[:i+1], "x", "F", "x", "F", "m", "N", "purple", gv, x_min, x_max, F_min, F_max)
        ]
        for idx, gi in enumerate(imgs): canvas[0:gv, idx*gv:(idx+1)*gv] = gi

        # テキスト
        txts = [f"x: {c['x']:.3f}m", f"v: {c['v']:.2f}m/s", f"a: {c['a']:.2f}m/s2", f"F: {c['F']:.3f}N"]
        for idx, tx in enumerate(txts):
            ts = cv2.getTextSize(tx, font_it, 0.8, 2)[0]
            cv2.putText(canvas, tx, (idx*gv + (gv-ts[0])//2, gv+60), font_it, 0.8, (255,255,255), 2)

        # 描画
        if pd.notna(c['gx']):
            cv2.circle(frame, (int(c['gx']), int(c['gy'])), mask_size, (150,150,150), 2)
            cv2.circle(frame, (int(c['gx']), int(c['gy'])), 6, (0,255,0), -1)
            if pd.notna(c['bx']):
                cv2.circle(frame, (int(c['bx']), int(c['by'])), 6, (255,0,255), -1)
                cv2.line(frame, (int(c['gx']),int(c['gy'])), (int(c['bx']),int(c['by'])), (255,255,255), 1)

        canvas[hh:, 0:w_orig] = frame
        t_str = f"t: {c['t']:.2f}s"
        t_sz = cv2.getTextSize(t_str, font_it, 1.1, 2)[0]
        cv2.putText(canvas, t_str, (w_orig-t_sz[0]-20, h_orig+hh-30), font_it, 1.1, (255,255,255), 2)
        out.write(canvas)
        if i % 20 == 0: progress_bar.progress(0.3 + (i / len(df)) * 0.7)

    cap_re.release()
    out.release()
    status.success("解析が完了しました！")
    
    st.download_button("📊 CSV保存", df[["t","x","v","a","F"]].to_csv(index=False).encode('utf_8_sig'), "kinema_cart_data.csv")
    with open(final_path, "rb") as f:
        st.download_button("🎥 解析動画保存", f, "cart_grapher_output.mp4")
    os.remove(tfile.name)
