import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import io, tempfile, os

# サーバー用設定
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="CartGrapher Debug")
st.title("🚀 CartGrapher Studio (Debug Mode)")

uploaded_file = st.file_uploader("動画を選択", type=["mp4", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # --- 安全な解析ループ ---
    data_log = []
    prog = st.progress(0.0)
    
    try:
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            # テスト用に時間と仮の座標を入れる（ここを後でトラッキングに差し替え）
            data_log.append({"t": i/fps, "x": np.sin(i/10)})
            
            # ValueError対策：進捗率を0.0~1.0の間に強制的に収める
            current_prog = min(max(i / total_frames, 0.0), 1.0)
            prog.progress(current_prog)
            
        cap.release()
        df = pd.DataFrame(data_log)

        if len(df) > 31: # データが十分にある時だけフィルタをかける
            df["x"] = savgol_filter(df["x"], 11, 2)
            df["v"] = df["x"].diff().fillna(0) * fps
        
        st.success("解析成功！")
        
        # グラフ表示
        fig, ax = plt.subplots()
        ax.plot(df["t"], df["x"])
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"解析中にエラーが発生しました: {e}")
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)
