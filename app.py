import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import tempfile
import os

# サーバーでの描画エラーを完全に防ぐ設定
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="CartGrapher")
st.title("🚀 CartGrapher Studio (Rescue Mode)")

# 1. 起動確認用のメッセージ
st.success("App is running! サーバーは正常に起動しています。")

# 2. サイドバー
radius_cm = st.sidebar.slider("半径(cm)", 0.5, 5.0, 1.6)
mass = st.sidebar.number_input("質量(kg)", value=0.1)

# 3. ファイルアップローダー
uploaded_file = st.file_uploader("動画を選択してください", type=["mp4", "mov"])

if uploaded_file is not None:
    # 読み込み中のクラッシュを防ぐため、非常にシンプルな処理に徹します
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None: fps = 30.0
    
    data = []
    # 最初の100フレームだけ解析（テスト用）
    for i in range(100):
        ret, frame = cap.read()
        if not ret: break
        # ここでは座標計算をせず、時間データだけ作成して動作テスト
        data.append({"t": i/fps, "x": np.sin(i/10)})
    
    cap.release()
    os.unlink(tfile.name) # 一時ファイルを確実に削除

    if data:
        df = pd.DataFrame(data)
        st.write("### 解析テスト結果")
        
        # グラフ作成（エラー防止のため try-except で囲む）
        try:
            fig, ax = plt.subplots()
            ax.plot(df["t"], df["x"])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Position (m)")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Graph Error: {e}")
        
        st.dataframe(df.head())
    else:
        st.error("動画を読み込めませんでした。")
