import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tempfile
import matplotlib.pyplot as plt
import io
import os

# --- 基本設定 ---
plt.switch_backend('Agg')
plt.rcParams['mathtext.fontset'] = 'cm'
RADIUS_M = 0.016
VERSION = "2.6.1"
MAX_DURATION = 10.0

def format_sci_latex(val):
    try:
        if abs(val) < 1e-6 and val != 0: return "0"
        s = f"{val:.2e}"
        base, exp = s.split('e')
        exp_int = int(exp)
        if exp_int == 0: return f"{float(base):.2f}"
        return rf"{base} \times 10^{{{exp_int}}}"
    except: return "0"

# 引数に markers を追加
def create_graph_image(df_sub, x_col, y_col, x_label, y_label, x_unit, y_unit, color, size, x_max, y_min, y_max, shade_range=None, markers=None):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    try:
        if not df_sub.empty:
            ax.plot(df_sub[x_col], df_sub[y_col], color=color, linewidth=2, alpha=0.8)
            # 現在値のドット
            ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=60, edgecolors='white', zorder=5)
            
            # 【追加】指定時刻にマーカー（オレンジ色の点）を打つ
            if markers is not None:
                for t_val in markers:
                    # 全データから該当時刻に最も近い行を特定（描画中の範囲外でもエラーにならないよう考慮）
                    # 描画対象のデータフレームから探す
                    m_row = df_sub.iloc[(df_sub['t']-t_val).abs().argsort()[:1]]
                    if not m_row.empty:
                        ax.scatter(m_row[x_col], m_row[y_col], color='orange', s=50, marker='o', edgecolors='black', zorder=10)

            if shade_range is not None and y_col == 'F':
                t_s, t_e = shade_range
                mask = (df_sub['t'] >= t_s) & (df_sub['t'] <= t_e)
                ax.fill_between(df_sub[x_col], df_sub[y_col], where=mask, color=color, alpha=0.3)
        
        ax.set_title(f"${y_label}$ - ${x_label}$", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"${x_label}$ [{x_unit}]", fontsize=11)
        ax.set_ylabel(f"${y_label}$ [{y_unit}]", fontsize=11)
        ax.set_xlim(0, max(float(x_max), 0.1))
        yr = max(float(y_max - y_min), 0.01)
        ax.set_ylim(y_min - yr*0.1, y_max + yr*0.1)
        ax.grid(True, linestyle='--', alpha=0.5)
    except: pass
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    return cv2.resize(img, (size, size)) if img is not None else np.zeros((size, size, 3), dtype=np.uint8)

st.set_page_config(page_title=f"CartGrapher Studio v{VERSION}", layout="wide")
st.title(f"🚀 CartGrapher Studio ver {VERSION}")

st.sidebar.header("解析設定")
mass_input = st.sidebar.number_input("台車の質量 $m$ [kg]", value=0.100, min_value=0.001, format="%.3f", step=0.001)
mask_size = st.sidebar.slider("解析エリア半径 [px]", 50, 400, 200, 10)

uploaded_file = st.file_uploader("動画をアップロード (10秒以内)", type=["mp4", "mov"])

if uploaded_file:
    tfile_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile_temp.write(uploaded_file.read())
    tfile_temp.close()

    if "df" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        with st.spinner("最高精度エンジンで解析中..."):
            cap = cv2.VideoCapture(tfile_temp.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            data_log = []; total_angle, prev_angle = 0.0, None; last_valid_gx, last_valid_gy = np.nan, np.nan
            L_G, L_P = (np.array([35,50,50]), np.array([85,255,255])), (np.array([140,40,40]), np.array([180,255,255]))
            f_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_g = cv2.inRange(hsv, L_G[0], L_G[1]); con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                new_gx, new_gy = np.nan, np.nan
                if con_g:
                    c = max(con_g, key=cv2.contourArea); M = cv2.moments(c)
                    if M["m00"] > 100: new_gx, new_gy = M["m10"]/M["m00"], M["m01"]/M["m00"]
                if not np.isnan(last_valid_gx) and not np.isnan(new_gx):
                    if np.sqrt((new_gx - last_valid_gx)**2 + (new_gy - last_valid_gy)**2) > 60: new_gx, new_gy = last_valid_gx, last_valid_gy
                gx = new_gx if not np.isnan(new_gx) else last_valid_gx; gy = new_gy if not np.isnan(new_gy) else last_valid_gy
                last_valid_gx, last_valid_gy = gx, gy
                bx, by = np.nan, np.nan
                if not np.isnan(gx):
                    mc = np.zeros((h, w), dtype=np.uint8); cv2.circle(mc, (int(gx), int(gy)), mask_size, 255, -1)
                    mask_p = cv2.inRange(cv2.bitwise_and(hsv, hsv, mask=mc), L_P[0], L_P[1]); con_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if con_p:
                        cp = max(con_p, key=cv2.contourArea); Mp = cv2.moments(cp)
                        if Mp["m00"]!=0: bx, by = Mp["m10"]/Mp["m00"], Mp["m01"]/Mp["m00"]
                if not np.isnan(gx) and not np.isnan(bx):
                    curr_a = np.arctan2(by - gy, bx - gx)
                    if prev_angle is not None:
                        diff = curr_a - prev_angle
                        if diff > np.pi: diff -= 2*np.pi
                        elif diff < -np.pi: diff += 2*np.pi
                        total_angle += diff
                    prev_angle = curr_a
                data_log.append({"t": f_idx/fps, "x": total_angle*RADIUS_M, "gx": gx, "gy": gy, "bx": bx, "by": by})
                f_idx += 1
            cap.release()
            df = pd.DataFrame(data_log).interpolate().ffill().bfill()
            if len(df) > 31:
                df["x"] = savgol_filter(df["x"], 15, 2); df["v"] = savgol_filter(df["x"].diff().fillna(0)*fps, 31, 2)
                df["a"] = savgol_filter(df["v"].diff().fillna(0)*fps, 31, 2); df["F"] = mass_input * df["a"]
            st.session_state.df = df; st.session_state.video_meta = {"fps": fps, "w": w, "h": h, "path": tfile_temp.name}; st.session_state.file_id = uploaded_file.name

    df = st.session_state.df
    st.sidebar.markdown("---")
    t_max_limit = float(df["t"].max())
    t1 = st.sidebar.number_input(r"開始時刻 $t_1$ [s]", 0.0, t_max_limit, 0.0, 0.01)
    row1 = df.iloc[(df['t']-t1).abs().argsort()[:1]]
    st.sidebar.latex(rf"x_1 = {row1['x'].values[0]:.3f} \,\, \mathrm{{m}}")
    st.sidebar.latex(rf"v_1 = {row1['v'].values[0]:.3f} \,\, \mathrm{{m/s}}")
    st.sidebar.markdown("---")
    t2 = st.sidebar.number_input(r"終了時刻 $t_2$ [s]", 0.0, t_max_limit, t_max_limit, 0.01)
    row2 = df.iloc[(df['t']-t2).abs().argsort()[:1]]
    st.sidebar.latex(rf"x_2 = {row2['x'].values[0]:.3f} \,\, \mathrm{{m}}")
    st.sidebar.latex(rf"v_2 = {row2['v'].values[0]:.3f} \,\, \mathrm{{m/s}}")

    time_list = [round(t, 4) for t in df["t"].tolist()]
    selected_t = st.select_slider("時刻をスキャン [s]", options=time_list, value=time_list[0])
    time_idx = time_list.index(selected_t); curr_row = df.iloc[time_idx]
    
    t_m, x_m = float(df["t"].max()), float(df["x"].max())
    v_mi, v_ma = float(df["v"].min()), float(df["v"].max())
    a_mi, a_ma = float(df["a"].min()), float(df["a"].max())
    f_mi, f_ma = float(df["F"].min()), float(df["F"].max())

    # マーカーとして渡すリスト
    marker_times = [t1, t2]

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.image(create_graph_image(df.iloc[:time_idx+1], "t", "x", "t", "x", "s", "m", 'blue', 450, t_m, 0.0, x_m, markers=marker_times), channels="BGR")
        st.latex(rf"x = {curr_row['x']:.3f} \,\, \mathrm{{m}}")
    with r1c2:
        st.image(create_graph_image(df.iloc[:time_idx+1], "t", "v", "t", "v", "s", "m/s", 'red', 450, t_m, v_mi, v_ma, markers=marker_times), channels="BGR")
        st.latex(rf"v = {curr_row['v']:.3f} \,\, \mathrm{{m/s}}")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.image(create_graph_image(df.iloc[:time_idx+1], "t", "a", "t", "a", "s", "m/s^2", 'green', 450, t_m, a_mi, a_ma, markers=marker_times), channels="BGR")
        st.latex(rf"a = {curr_row['a']:.3f} \,\, \mathrm{{m/s^2}}")
    with r2c2:
        st.image(create_graph_image(df.iloc[:time_idx+1], "x", "F", "x", "F", "m", "N", 'purple', 450, x_m, f_mi, f_ma, shade_range=(t1, t2), markers=marker_times), channels="BGR")
        st.latex(rf"F = {curr_row['F']:.3f} \,\, \mathrm{{N}}")

    st.divider()
    df_w = df[(df["t"] >= t1) & (df["t"] <= t2)]
    if len(df_w) > 1:
        w_val = np.trapz(df_w["F"], df_w["x"])
        st.latex(rf"W = {format_sci_latex(w_val)} \,\, \mathrm{{J}}")

    if st.button(f"🎥 解析動画を生成して保存"):
        meta = st.session_state.video_meta
        final_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        v_size = meta["w"] // 4
        header_h = v_size + 100
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        graph_configs = [
            {"xc": "t", "yc": "x", "xl": "t", "yl": "x", "xu": "s", "yu": "m", "col": "blue", "ymn": 0.0, "ymx": x_m, "xm": t_m},
            {"xc": "t", "yc": "v", "xl": "t", "yl": "v", "xu": "s", "yu": "m/s", "col": "red", "ymn": v_mi, "ymx": v_ma, "xm": t_m},
            {"xc": "t", "yc": "a", "xl": "t", "yl": "a", "xu": "s", "yu": "m/s^2", "col": "green", "ymn": a_mi, "ymx": a_ma, "xm": t_m},
            {"xc": "x", "yc": "F", "xl": "x", "yl": "F", "xu": "m", "yu": "N", "col": "purple", "ymn": f_mi, "ymx": f_ma, "xm": x_m}
        ]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(final_path, fourcc, meta["fps"], (meta["w"], meta["h"] + header_h))
        
        cap = cv2.VideoCapture(meta["path"])
        p_bar = st.progress(0.0)
        status_text = st.empty()
        
        for i in range(len(df)):
            ret, frame = cap.read()
            if not ret: break
            canvas = np.zeros((meta["h"] + header_h, meta["w"], 3), dtype=np.uint8)
            curr = df.iloc[i]
            df_s = df.iloc[:i+1]
            
            for idx, g in enumerate(graph_configs):
                # 動画内グラフにもマーカーを表示
                g_img = create_graph_image(df_s, g["xc"], g["yc"], g["xl"], g["yl"], g["xu"], g["yu"], g["col"], v_size, g["xm"], g["ymn"], g["ymx"], shade_range=(t1, t2) if g["yc"]=="F" else None, markers=marker_times)
                canvas[0:v_size, idx*v_size:(idx+1)*v_size] = g_img
                val_text = f"{g['yl']} = {curr[g['yc']]:>+7.3f} {g['yu']}"
                (tw, th), _ = cv2.getTextSize(val_text, font, 0.5, 1)
                cv2.putText(canvas, val_text, (idx*v_size + (v_size-tw)//2, v_size + 50), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            t_text = f"t = {curr['t']:.2f} s"
            cv2.putText(frame, t_text, (20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
            
            if not np.isnan(curr['gx']):
                cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), mask_size, (255,255,0), 2)
                cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), 5, (0,255,0), -1)
                if not np.isnan(curr['bx']):
                    cv2.circle(frame, (int(curr['bx']), int(curr['by'])), 5, (255,0,255), -1)
            
            canvas[header_h:, :] = frame
            out.write(canvas)
            if i % 10 == 0:
                p_bar.progress(i / len(df))
                status_text.text(f"生成中: {i}/{len(df)} フレーム")

        cap.release()
        out.release()
        p_bar.empty()
        status_text.success("✅ 動画生成が完了しました！")
        
        with open(final_path, "rb") as f:
            st.download_button("💾 完成した動画をダウンロード", f, file_name=f"analysis_v{VERSION}.mp4")
