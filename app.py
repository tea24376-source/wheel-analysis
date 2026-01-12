import streamlit as st
# サーバー環境でのエラーを防ぐため、一番最初に Matplotlib のバックエンドを固定
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tempfile
import os
import io

# --- ページ基本設定（エラー防止のため関数の外で実行） ---
st.set_page_config(page_title="CartGrapher Studio", layout="wide")

# セッション状態の初期化（これをしないと、アクセスした瞬間に変数がなくて落ちることがあります）
if "df" not in st.session_state:
    st.session_state.df = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

st.title("🚀 CartGrapher Studio")

# --- グラフ生成関数 ---
def create_graph_image(df_sub, x_col, y_col, x_label, y_label, x_unit, y_unit, color, size, x_min, x_max, y_min, y_max):
    try:
        fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
        if not df_sub.empty:
            ax.plot(df_sub[x_col].values, df_sub[y_col].values, color=color, linewidth=2)
            ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=50)
        
        ax.set_title(f"${y_label}$ - ${x_label}$", fontsize=16, fontweight='bold')
        ax.set_xlabel(f"${x_label}$ [{x_unit}]", fontsize=14)
        ax.set_ylabel(f"${y_label}$ [{y_unit}]", fontsize=14)
        
        xr = max(float(x_max - x_min), 0.001)
        yr = max(float(y_max - y_min), 0.001)
        ax.set_xlim(float(x_min) - xr*0.05, float(x_max) + xr*0.05)
        ax.set_ylim(float(y_min) - yr*0.1, float(y_max) + yr*0.1)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor='white')
        plt.close(fig)
        buf.seek(0)
        img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
        return cv2.resize(img, (size, size))
    except:
        return np.zeros((size, size, 3), dtype=np.uint8)

# --- メインロジック（サイドバー） ---
st.sidebar.header("Kinema-Cart 設定")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mass = st.sidebar.number_input("台車の質量 $m$ (kg)", value=0.100, min_value=0.001, format="%.3f")
mask_size = st.sidebar.slider("解析エリア半径 (px)", 50, 400, 200, 10)

uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov"])

# 動画がない時は案内を表示して終了（エラーを未然に防ぐ）
if uploaded_file is None:
    st.info("💡 上のボタンから Kinema-Cart の実験動画をアップロードしてください。")
    st.stop() 

# --- 以下、前回の高速解析ロジック ---
# （ここからは uploaded_file がある時だけ実行されるので安全です）
