import streamlit as st
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import datetime
import pytz

# 어플 기본 설정
st.set_page_config(page_title="나만의 주식 대시보드", layout="wide")

# 한글 폰트 적용 (클라우드 환경용)
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
fm.fontManager.addfont(fontpath)
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False

# 17시 기준 날짜 세팅
kst = pytz.timezone('Asia/Seoul')
now_kst = datetime.datetime.now(kst)
if now_kst.hour < 17:
    cutoff_date = (now_kst - datetime.timedelta(days=1)).date()
else:
    cutoff_date = now_kst.date()

cutoff_str = cutoff_date.strftime("%Y-%m-%d")
weekdays = ["월", "화", "수", "목", "금", "토", "일"]
formatted_date = cutoff_date.strftime("%m%d") + f"({weekdays[cutoff_date.weekday()]})"

ASSET_DICT = {
    "KOSPI": "^KS11", "KODEX 200타겟위클리커버드콜": "498400.KS",
    "TIGER배당커버드콜액티브": "472150.KS", "나스닥100": "^NDX",
    "TIME미국나스닥100액티브": "426030.KS", "금 현물 (달러 기준)": "GC=F",
    "금 현물 (원화 기준)": "GOLD_KRW", "은 현물 (달러 기준)": "SI=F"
}

@st.cache_data(ttl=3600)
def get_historical_data(ticker, period, interval):
    if ticker == "GOLD_KRW":
        gold = yf.Ticker("GC=F").history(period=period, interval=interval)
        usdkrw = yf.Ticker("KRW=X").history(period=period, interval=interval)
        
        # 야후파이낸스 MultiIndex 에러 방지
        if isinstance(gold.columns, pd.MultiIndex): gold.columns = gold.columns.get_level_values(0)
        if isinstance(usdkrw.columns, pd.MultiIndex): usdkrw.columns = usdkrw.columns.get_level_values(0)
            
        if gold.empty or usdkrw.empty: return pd.DataFrame()
        if getattr(gold.index, 'tz', None) is not None: gold.index = gold.index.tz_localize(None)
        if getattr(usdkrw.index, 'tz', None) is not None: usdkrw.index = usdkrw.index.tz_localize(None)
        usdkrw = usdkrw.reindex(gold.index, method='ffill')
        df = gold.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = (gold[col] * usdkrw['Close']) / 31.1034768
        df = df.dropna()
    else:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return df
        if getattr(df.index, 'tz', None) is not None: df.index = df.index.tz_localize(None)
    df = df[df.index.normalize() <= pd.to_datetime(cutoff_str)]
    return df

def add_indicators(df, ma_list):
    if len(df) == 0: return df
    for ma in ma_list: df[f'MA_{ma}'] = df['Close'].rolling(window=ma).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df[f'MA_{ma_list[0]}'] + (std_20 * 2)
    df['BB_Lower'] = df[f'MA_{ma_list[0]}'] - (std_20 * 2)
    
    tenkan = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    kijun = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    senkou_a_un = (tenkan + kijun) / 2
    senkou_b_un = (df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2
    df['Tenkan'], df['Kijun'] = tenkan, kijun
    
    if len(df) >= 2:
        last_date = df.index[-1]
        diff = df.index[-1] - df.index[-2]
        future_idx = [last_date + diff * i for i in range(1, 27)]
        future_df = pd.DataFrame(index=future_idx, columns=df.columns)
        
        df_ext = pd.concat([df, future_df])
        
        # [핵심 에러 해결] 모든 데이터를 강제로 숫자형(float)으로 변환
        for col in df_ext.columns:
            df_ext[col] = pd.to_numeric(df_ext[col], errors='coerce')
            
        df_ext['Senkou_A'] = senkou_a_un.reindex(df_ext.index).shift(26)
        df_ext['Senkou_B'] = senkou_b_un.reindex(df_ext.index).shift(26)
        return df_ext
    return df

def draw_chart(df, ax, title, ma_list):
    if len(df.dropna(subset=['Close'])) < 2: return
    apds = []
    if df['Tenkan'].notna().any(): apds.append(mpf.make_addplot(df['Tenkan'], ax=ax, color='#00FF00', width=1, alpha=0.5))
    if df['Kijun'].notna().any(): apds.append(mpf.make_addplot(df['Kijun'], ax=ax, color='#8A2BE2', width=1, alpha=0.5))
        
    ma_colors = ['#FFA500', '#1E90FF', '#8B4513', '#000000']
    for ma, color in zip(ma_list, ma_colors):
        if df[f'MA_{ma}'].notna().any(): apds.append(mpf.make_addplot(df[f'MA_{ma}'], ax=ax, color=color, width=1.2, alpha=0.8))
            
    if df['BB_Upper'].notna().any(): apds.append(mpf.make_addplot(df['BB_Upper'], ax=ax, color='gray', linestyle=':', width=2.0, alpha=0.8))
    if df['BB_Lower'].notna().any(): apds.append(mpf.make_addplot(df['BB_Lower'], ax=ax, color='gray', linestyle=':', width=2.0, alpha=0.8))
            
    mpf.plot(df, type='candle', ax=ax, addplot=apds, ylabel='')
    
    ax.text(0.02, 0.97, f"{formatted_date} 17시 기준\n{title}", transform=ax.transAxes, ha='left', va='top', fontsize=22, fontweight='heavy', color='#333333', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'), zorder=10)
    
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Price", fontweight='bold', fontsize=14)
    ax.tick_params(axis='y', labelsize=18)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    
    x_idx = np.arange(len(df))
    valid = df['Senkou_A'].notna() & df['Senkou_B'].notna()
    if valid.any():
        ax.fill_between(x_idx[valid], df['Senkou_A'][valid], df['Senkou_B'][valid], where=df['Senkou_A'][valid] >= df['Senkou_B'][valid], facecolor='lightcoral', alpha=0.3)
        ax.fill_between(x_idx[valid], df['Senkou_A'][valid], df['Senkou_B'][valid], where=df['Senkou_A'][valid] < df['Senkou_B'][valid], facecolor='lightblue', alpha=0.3)

    real_data = df.dropna(subset=['Close'])
    current_close = real_data['Close'].iloc[-1]
    prev_close = real_data['Close'].iloc[-2]
    current_idx = len(real_data) - 1
    
    ax.hlines(current_close, xmin=0, xmax=len(df), colors='black', linestyles='--', linewidth=1.5, alpha=0.7)
    
    max_high = real_data['High'].max()
    diff_from_high = (max_high - current_close) / current_close * 100
    change = current_close - prev_close
    change_pct = (change / prev_close) * 100
    
    table_data, cell_colors = [], []
    change_pct_int = int(change_pct)
    if change_pct_int > 0: change_pct_str = f"{change_pct_int:+d}%▲"
    elif change_pct_int < 0: change_pct_str = f"{change_pct_int:+d}%▼"
    else: change_pct_str = f"{change_pct_int}%"
    
    table_data.append(["종가", f"{int(current_close):,d}", change_pct_str])
    cell_colors.append(["#f0f0f0", "#ffffff", "#ffffff"])
    
    diff_high_int = int(diff_from_high)
    if diff_high_int > 0: diff_high_str = f"{diff_high_int:+d}%▲"
    elif diff_high_int < 0: diff_high_str = f"{diff_high_int:+d}%▼"
    else: diff_high_str = f"{diff_high_int}%"
    
    abs_diff_high = abs(diff_high_int)
    if abs_diff_high <= 5: bg_color_high = "#FFC0CB" 
    elif abs_diff_high <= 10: bg_color_high = "#FFFF00" 
    else: bg_color_high = "#ffffff"
    
    table_data.append(["전고점", f"{int(max_high):,d}", diff_high_str])
    cell_colors.append(["#f0f0f0", "#ffffff", bg_color_high])
    
    for ma, color in zip(ma_list, ma_colors):
        ma_val = real_data[f'MA_{ma}'].iloc[-1]
        if not np.isnan(ma_val):
            diff = int((ma_val - current_close) / current_close * 100)
            if diff > 0: diff_str = f"{diff:+d}%▲"
            elif diff < 0: diff_str = f"{diff:+d}%▼"
            else: diff_str = f"{diff}%"
            
            abs_diff = abs(diff)
            if abs_diff <= 5: bg_color = "#FFC0CB"
            elif abs_diff <= 10: bg_color = "#FFFF00"
            else: bg_color = "#ffffff"
            
            table_data.append([f"{ma}MA", f"{int(ma_val):,d}", diff_str])
            cell_colors.append(["#f0f0f0", "#ffffff", bg_color])
        else:
            table_data.append([f"{ma}MA", "-", "-"])
            cell_colors.append(["#f0f0f0", "#ffffff", "#ffffff"])
            
    table = ax.table(cellText=table_data, cellColours=cell_colors, loc='upper left', bbox=[0.02, 0.40, 0.45, 0.45], cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.set_zorder(10)
    
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('gray')
        if j == 0: cell.set_text_props(weight='bold')
        if j == 2 and i > 0: cell.set_text_props(weight='bold')
    
    ymin, ymax = ax.get_ylim()
    min_gap = (ymax - ymin) * 0.10
    
    right_labels = []
    highlight_text = f"{int(current_close):,d}\n{int(change):+d} ({int(change_pct):+d}%)"
    right_labels.append({'text': highlight_text, 'val': current_close, 'target_y': current_close, 'x_offset': 2, 'color': 'black', 'text_color': 'black', 'bg_color': 'white'})
    
    for ma, color in zip(ma_list, ma_colors):
        ma_val = real_data[f'MA_{ma}'].iloc[-1]
        if not np.isnan(ma_val):
            diff = int((ma_val - current_close) / current_close * 100)
            right_labels.append({'text': f"{ma}MA: {int(ma_val):,d}\n({diff:+d}%)", 'val': ma_val, 'target_y': ma_val, 'x_offset': 14, 'color': color, 'text_color': color, 'bg_color': 'white'})
            
    right_labels.sort(key=lambda x: x['val'], reverse=True)
    
    for _ in range(15):
        for i in range(len(right_labels) - 1):
            diff_y = right_labels[i]['target_y'] - right_labels[i+1]['target_y']
            if diff_y < min_gap:
                overlap = min_gap - diff_y
                right_labels[i]['target_y'] += overlap / 2
                right_labels[i+1]['target_y'] -= overlap / 2

    for item in right_labels:
        bbox_props = dict(boxstyle="round,pad=0.4", fc='white', ec='black', lw=2.0, alpha=0.9) if item['color'] == 'black' else dict(boxstyle="round,pad=0.2", fc=item['bg_color'], alpha=0.8, edgecolor=item['color'])
        ax.annotate(item['text'], xy=(current_idx, item['val']), xycoords='data', xytext=(current_idx + item['x_offset'], item['target_y']), textcoords='data', ha='left', va='center', color=item['text_color'], fontsize=10, fontweight='bold', bbox=bbox_props, arrowprops=dict(arrowstyle='-', color=item['color'], alpha=0.6), clip_on=False)

    max_idx_date = real_data['High'].idxmax()
    max_idx_int = df.index.get_loc(max_idx_date)
    if isinstance(max_idx_int, slice): max_idx_int = max_idx_int.start
    elif getattr(max_idx_int, "shape", None): max_idx_int = np.where(max_idx_int)[0][0]
        
    all_y_values = [item['target_y'] for item in right_labels] + [max_high, ymin, ymax]
    margin = (max(all_y_values) - min(all_y_values)) * 0.25
    ax.set_ylim(min(all_y_values) - margin, max(all_y_values) + margin * 1.8)
    
    ax.annotate(f"전고점: {int(max_high):,d}\n(필요 상승률: {int(diff_from_high):+d}%)", xy=(max_idx_int, max_high), xycoords='data', xytext=(0, 25), textcoords='offset points', ha='center', va='bottom', color='#d62728', fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", lw=1.5, alpha=0.9), arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5), clip_on=False)

# 어플 화면 UI (디자인)
st.title("📊 나만의 주식 자동 분석 어플")
selected_asset = st.selectbox("어떤 종목을 확인하시겠습니까?", list(ASSET_DICT.keys()))

if st.button("차트 그리기"):
    with st.spinner('실시간 데이터를 분석하여 차트를 그리고 있습니다...'):
        ticker = ASSET_DICT[selected_asset]
        df_full_daily = get_historical_data(ticker, period="5y", interval="1d")
        
        if df_full_daily.empty:
            st.error("데이터를 불러오지 못했습니다.")
        else:
            df_full_weekly = get_historical_data(ticker, period="10y", interval="1wk")
            df_full_monthly = get_historical_data(ticker, period="25y", interval="1mo")
            
            ma_list = [20, 60, 120, 200]
            df_daily_ind = add_indicators(df_full_daily, ma_list)
            df_weekly_ind = add_indicators(df_full_weekly, ma_list)
            df_monthly_ind = add_indicators(df_full_monthly, ma_list)

            df_daily = df_daily_ind.iloc[-(250+26):] if len(df_daily_ind) > (250+26) else df_daily_ind
            df_weekly = df_weekly_ind.iloc[-(104+26):] if len(df_weekly_ind) > (104+26) else df_weekly_ind
            df_monthly = df_monthly_ind.iloc[-(36+26):] if len(df_monthly_ind) > (36+26) else df_monthly_ind

            mc = mpf.make_marketcolors(up='r', down='b', edge='inherit', wick='inherit')
            s  = mpf.make_mpf_style(marketcolors=mc, rc={'font.family': 'NanumBarunGothic', 'axes.unicode_minus': False})

            fig = mpf.figure(figsize=(10, 18), style=s)
            ax1 = fig.add_subplot(3,1,1)
            ax2 = fig.add_subplot(3,1,2)
            ax3 = fig.add_subplot(3,1,3)

            draw_chart(df_daily, ax1, f"{selected_asset} (일봉)", ma_list)
            draw_chart(df_weekly, ax2, f"{selected_asset} (주봉)", ma_list)
            draw_chart(df_monthly, ax3, f"{selected_asset} (월봉)", ma_list)

            plt.tight_layout()
            st.pyplot(fig)
