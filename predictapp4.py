import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta
from lightgbm import LGBMRegressor
import plotly.graph_objects as go
import jpholiday

# --- 設定 ---
st.set_page_config(page_title="全エリア インバランス料金予測システム", layout="wide")

AREAS = ["北海道", "東北", "東京", "中部", "北陸", "関西", "中国", "四国", "九州"]
AREA_COORDS = {
    "北海道": (43.0642, 141.3469), "東北": (38.2682, 140.8694),
    "東京": (35.6895, 139.6917), "中部": (35.1815, 136.9064),
    "北陸": (36.5947, 136.6256), "関西": (34.6937, 135.5023),
    "中国": (34.3853, 132.4553), "四国": (34.3401, 134.0434),
    "九州": (33.5902, 130.4017)
}

# --- データ取得関数 ---

@st.cache_data(ttl=3600)
def get_imbalance_data(target_month):
    """APIからインバランス料金を取得し、整形する"""
    url = f"https://www.imbalanceprices-cs.jp/api/1.0/imb/price/{target_month}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200:
            csv_text = response.content.decode('cp932')
            lines = csv_text.split('\n')
            header_idx = next(i for i, line in enumerate(lines) if line.count(',') > 10)
            
            df = pd.read_csv(io.StringIO(csv_text), skiprows=header_idx + 1)
            
            if "Unnamed: 22" in df.columns:
                df = df.loc[:, :"Unnamed: 22"]
            
            rename_dict = {}
            has_date, has_time = False, False
            for col in df.columns:
                if not has_date and any(kw in col for kw in ['受渡日', '対象日', '日付', '年月日']):
                    rename_dict[col] = 'Date'
                    has_date = True
                if not has_time and any(kw in col for kw in ['時刻', 'コマ']):
                    rename_dict[col] = 'Time'
                    has_time = True
            
            df.rename(columns=rename_dict, inplace=True)
            df.columns = [col.replace('エリア', '') for col in df.columns]
            
            if 'Date' in df.columns and 'Time' in df.columns:
                df['Date'] = pd.to_numeric(df['Date'], errors='coerce')
                df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
                df = df.dropna(subset=['Date', 'Time'])
                
                df['Date'] = df['Date'].astype(int).astype(str)
                df['Time'] = df['Time'].astype(int)
                
                base_date = pd.to_datetime(df['Date'], format='%Y%m%d')
                time_delta = pd.to_timedelta((df['Time'] - 1) * 30, unit='m')
                df['Datetime'] = base_date + time_delta
                df = df.set_index('Datetime')
                
                return df[AREAS].apply(pd.to_numeric, errors='coerce')
        return None
    except Exception as e:
        st.error(f"インバランスデータ取得エラー ({target_month}): {e}")
        return None

@st.cache_data(ttl=3600)
def get_weather_data(start_date, end_date):
    """Open-Meteoから全エリアの気象データを取得し補間する"""
    lats = ",".join([str(coords[0]) for coords in AREA_COORDS.values()])
    lons = ",".join([str(coords[1]) for coords in AREA_COORDS.values()])
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}&start_date={start_date.date()}&end_date={end_date.date()}&hourly=temperature_2m,shortwave_radiation,precipitation&timezone=Asia%2FTokyo"
    
    try:
        res = requests.get(url, timeout=20)
        # 【修正】エラーの理由をしっかり画面に表示するように変更
        if res.status_code == 200:
            data = res.json()
            all_dfs = []
            for i, area in enumerate(AREAS):
                area_data = data[i] if isinstance(data, list) else data
                temp_df = pd.DataFrame({
                    "Datetime": pd.to_datetime(area_data["hourly"]["time"]),
                    f"{area}_気温": area_data["hourly"]["temperature_2m"],
                    f"{area}_日射量": area_data["hourly"]["shortwave_radiation"],
                    f"{area}_降水量": area_data["hourly"]["precipitation"]
                })
                all_dfs.append(temp_df.set_index("Datetime"))
            
            combined = pd.concat(all_dfs, axis=1)
            
            df_res = combined.resample('30min').asfreq()
            for area in AREAS:
                df_res[f"{area}_気温"] = df_res[f"{area}_気温"].interpolate(method='linear')
                df_res[f"{area}_日射量"] = df_res[f"{area}_日射量"].interpolate(method='linear')
                df_res[f"{area}_降水量"] = df_res[f"{area}_降水量"].ffill() / 2.0
                
            return df_res
        else:
            st.error(f"⚠️ 気象APIエラー ({res.status_code}): {res.text}")
            return None
    except Exception as e:
        st.error(f"⚠️ 気象データ通信エラー: {e}")
        return None

@st.cache_data(ttl=3600)
def get_spot_data():
    """ローカルのCSVからJEPXスポット価格を取得し、インバランスと同じ形式に整形する"""
    file_path = "spot_summary_2025.csv"
    try:
        df = pd.read_csv(file_path, encoding='cp932')
        
        df['Date'] = pd.to_datetime(df['受渡日'])
        time_delta = pd.to_timedelta((df['時刻コード'] - 1) * 30, unit='m')
        df['Datetime'] = df['Date'] + time_delta
        df = df.set_index('Datetime')
        
        rename_dict = {
            'エリアプライス北海道(円/kWh)': '北海道_スポット',
            'エリアプライス東北(円/kWh)': '東北_スポット',
            'エリアプライス東京(円/kWh)': '東京_スポット',
            'エリアプライス中部(円/kWh)': '中部_スポット',
            'エリアプライス北陸(円/kWh)': '北陸_スポット',
            'エリアプライス関西(円/kWh)': '関西_スポット',
            'エリアプライス中国(円/kWh)': '中国_スポット',
            'エリアプライス四国(円/kWh)': '四国_スポット',
            'エリアプライス九州(円/kWh)': '九州_スポット'
        }
        df = df.rename(columns=rename_dict)
        
        spot_cols = list(rename_dict.values())
        df = df[spot_cols]
        df = df[~df.index.duplicated(keep='first')].sort_index()
        
        return df.apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        st.error(f"スポットデータ取得エラー。パスやファイル名を確認してください: {e}")
        return None

def is_dayoff(date_obj):
    return date_obj.weekday() >= 5 or jpholiday.is_holiday(date_obj)

# --- アプリケーションUI ---

st.title("⚡ Imbalance Predictor Pro")
st.write("対象日の直近30日間の全エリア気象・スポット価格データから、各エリアの30分コマ別インバランス料金を予測し、詳細なAI分析レポートを提供します。")

st.sidebar.header("予測パラメータ設定")
predict_date = st.sidebar.date_input("予測/バックテスト対象日を入力", value=datetime.today().date() - timedelta(days=1))
predict_dt = datetime.combine(predict_date, datetime.min.time())

day_type_filter = st.sidebar.radio(
    "学習データのフィルタリング",
    ("フィルタなし (過去30日の全データを使用)", "平日/休日で分ける (予測日と同じ区分のみ使用)")
)

if st.sidebar.button("予測を実行する"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    train_start = predict_dt - timedelta(days=30)
    
    months_to_fetch = pd.date_range(start=train_start, end=predict_dt, freq='MS').strftime("%Y%m").tolist()
    if train_start.strftime("%Y%m") not in months_to_fetch:
        months_to_fetch.insert(0, train_start.strftime("%Y%m"))
    if predict_dt.strftime("%Y%m") not in months_to_fetch:
        months_to_fetch.append(predict_dt.strftime("%Y%m"))
    months_to_fetch = sorted(list(set(months_to_fetch)))

    status_text.text("1/5: インバランス料金(過去30日分)を取得中...")
    imb_list = [get_imbalance_data(m) for m in months_to_fetch]
    
    valid_imb_list = [df for df in imb_list if df is not None]
    if not valid_imb_list:
        st.error("インバランスデータの取得に失敗しました。対象月のデータがまだ公表されていない可能性があります。")
        progress_bar.empty()
        status_text.empty()
        st.stop()
        
    imb_df = pd.concat(valid_imb_list)
    imb_df = imb_df[~imb_df.index.duplicated(keep='first')].sort_index()
    progress_bar.progress(20)

    status_text.text("2/5: 全エリア気象データを取得中...")
    weather_df = get_weather_data(train_start, predict_dt + timedelta(days=1))
    progress_bar.progress(40)

    status_text.text("3/5: JEPXスポット価格データを取得中...")
    spot_df = get_spot_data()
    progress_bar.progress(60)

    # 【修正】どこが原因で止まったのか、明確にエラーメッセージを出す
    if imb_df.empty:
        st.error("❌ インバランスデータの取得結果が空です。対象月のデータが存在しない可能性があります。")
    elif weather_df is None:
        st.error("❌ 気象データ（Open-Meteo API）の取得に失敗しました。画面上部の赤いエラーメッセージを確認してください。")
    elif spot_df is None:
        st.error("❌ スポット価格(CSV)の読み込みに失敗しました。")
    else:
        features_df = pd.concat([weather_df, spot_df], axis=1).dropna()
        master_df = pd.concat([imb_df, features_df], axis=1).dropna()
        
        master_df['hour'] = master_df.index.hour
        master_df['minute'] = master_df.index.minute
        master_df['dayofweek'] = master_df.index.dayofweek
        master_df['is_dayoff'] = master_df.index.map(lambda x: int(is_dayoff(x.date()))) 
        
        train_data = master_df[master_df.index < predict_dt]
        
        target_is_dayoff = is_dayoff(predict_dt.date())
        
        if day_type_filter == "平日/休日で分ける (予測日と同じ区分のみ使用)":
            if target_is_dayoff:
                train_data = train_data[train_data['is_dayoff'] == 1]
                status_text.text("4/5: 【休日(土日祝)】の過去データからモデルを学習中...")
            else:
                train_data = train_data[train_data['is_dayoff'] == 0]
                status_text.text("4/5: 【平日】の過去データからモデルを学習中...")
        else:
            status_text.text("4/5: 全データを使用してモデルを学習中...")
            
        target_features = features_df[(features_df.index >= predict_dt) & (features_df.index < predict_dt + timedelta(days=1))].copy()
        
        if target_features.empty:
             st.error("予測対象日の特徴量（気象またはスポット価格）が存在しません。CSVファイルが最新か確認してください。")
             progress_bar.empty()
             status_text.empty()
             st.stop()
             
        target_features['hour'] = target_features.index.hour
        target_features['minute'] = target_features.index.minute
        target_features['dayofweek'] = target_features.index.dayofweek
        target_features['is_dayoff'] = int(target_is_dayoff)
        
        predictions = {}
        reports = {}
        
        all_features = list(features_df.columns) + ["hour", "minute", "dayofweek", "is_dayoff"]
        
        for area in AREAS:
            X_train = train_data[all_features]
            y_train = train_data[area]
            
            model = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train)
            
            X_target = target_features[all_features]
            pred = model.predict(X_target)
            predictions[area] = pred

            mean_spot = X_target[f"{area}_スポット"].mean()
            max_spot = X_target[f"{area}_スポット"].max()
            min_spot = X_target[f"{area}_スポット"].min()
            
            mean_temp = X_target[f"{area}_気温"].mean()
            max_temp = X_target[f"{area}_気温"].max()
            min_temp = X_target[f"{area}_気温"].min()
            mean_solar = X_target[f"{area}_日射量"].mean()
            mean_precip = X_target[f"{area}_降水量"].mean()
            
            mean_pred = pred.mean()
            max_pred = pred.max()
            min_pred = pred.min()
            
            max_pred_idx = np.argmax(pred)
            max_pred_time = target_features.index[max_pred_idx].strftime('%H:%M')
            
            mean_past_imb = y_train.mean()
            max_past_imb = y_train.max()
            mean_past_temp = X_train[f"{area}_気温"].mean()
            mean_past_spot = X_train[f"{area}_スポット"].mean()
            
            temp_diff = mean_temp - mean_past_temp
            spot_diff = mean_spot - mean_past_spot
            
            spot_rel = "上回る" if mean_pred > mean_spot else "下回る"
            day_type_text = "休日（土日・祝日）" if target_is_dayoff else "平日"
            
            if mean_temp > 25 or max_temp > 30:
                temp_eval = "気温が高く、冷房需要の大幅な増加が見込まれます。電力需給の逼迫リスクが高まりやすい環境です。"
            elif mean_temp < 10 or min_temp < 5:
                temp_eval = "厳しい寒さにより、暖房需要の急増による電力消費量の増加が想定されます。"
            else:
                temp_eval = "気候は比較的穏やかであり、空調需要による極端な電力消費の変動は限定的と考えられます。"
                
            if mean_solar > 150:
                solar_eval = "日中は豊富な日射量により太陽光発電の積極的な稼働が期待され、昼間帯のインバランス価格の押し下げ効果が強く働くでしょう。"
            elif mean_solar < 50:
                solar_eval = "日射量が少なく太陽光発電の出力が低迷するため、昼間帯でも火力発電等への依存度が高まり、価格が下がりにくい状況です。"
            else:
                solar_eval = "太陽光発電は標準的な出力が見込まれますが、天候の急変による需給バランスのブレに注意が必要です。"

            spread = mean_pred - mean_spot
            if spread > 3.0:
                spread_eval = "スポット価格に対してインバランス料金が明確に上振れする（プレミアムがつく）強いシグナルが出ています。不足インバランスには多大なペナルティが発生する危険な状態のため、事前の調達を厚めにすることを推奨します。"
            elif spread < -3.0:
                spread_eval = "スポット価格に対してインバランス料金が下振れする（ディスカウントされる）予測です。余剰インバランスによるネガティブな影響や、市場での売り逃しに警戒が必要です。"
            else:
                spread_eval = "インバランス料金は概ねスポット価格に連動した推移を見せており、極端なプレミアムやディスカウントは発生しにくい安定した相場環境と予測されます。"

            report_text = f"""
💡 **【AI予測詳細レポート（{area}エリア）】**

**1. 本日の気象予報と電力需給への影響**
予測対象日（{day_type_text}）の{area}エリアの気象予報について、平均気温は{mean_temp:.1f}℃（最高{max_temp:.1f}℃ / 最低{min_temp:.1f}℃）となっており、過去30日間の平均（{mean_past_temp:.1f}℃）と比較して{temp_diff:+.1f}℃の変化を示しています。{temp_eval}
また、平均日射量は{mean_solar:.1f} W/m²、平均降水量は{mean_precip:.1f} mm/hと予測されています。{solar_eval}
電力需要は気温および天候と密接に連動するため、本日の気象条件はインバランス価格を決定する上で非常に重要なベースラインとなります。

**2. JEPX前日市場（スポット）価格の動向**
本日のJEPXスポット価格は、1日を通じて平均{mean_spot:.1f}円/kWhで推移し、最高値は{max_spot:.1f}円/kWh、最安値は{min_spot:.1f}円/kWhと約定されています。過去30日間の平均スポット価格（{mean_past_spot:.1f}円/kWh）と比較すると{spot_diff:+.1f}円の違いがあります。
インバランス精算単価は基本的にこのスポット価格を基準（アンカー）として決定されるため、スポット価格が高い時間帯はインバランス価格も高騰しやすい強い相関関係を持ちます。AIモデルは、このスポット価格の形状と気象データを掛け合わせることで、基準値からの「乖離幅（アルファ）」を計算しています。

**3. 過去30日間のインバランストレンドと本日の予測詳細**
{area}エリアにおける過去30日間のインバランス料金の実績を振り返ると、平均で{mean_past_imb:.1f}円/kWh、最大で{max_past_imb:.1f}円/kWhのスパイクを記録していました。
これらの過去の傾向と、本日の全エリアの気象予報・スポット価格を統合的にLightGBMモデルで解析した結果、本日のインバランス料金は **平均 {mean_pred:.1f}円/kWh** と予測されました。
予測の最高値は **{max_pred:.1f}円/kWh** （発生予測時刻：**{max_pred_time}**頃）、最安値は **{min_pred:.1f}円/kWh** となっています。全体的なトレンドとして、本日はスポット価格を**{spot_rel}**水準で推移する可能性が高いとモデルは判断しています。

**4. 総合評価とリスク要因**
{spread_eval}
本モデルは、対象エリアの局所的なデータだけでなく、他エリア（北海道から九州まで）の気象および市場データも同時に学習しています。これにより、広域的な電力融通（連系線の運用）による影響や、他エリアでの天候急変がもたらす価格の波及効果も織り込んだ予測を行っています。
特に **{max_pred_time}** 前後は1日の中で最も価格が高騰するリスクが高い時間帯と予測されているため、ポジションの傾き（不足インバランスの発生）には十分な注意を払うことを推奨します。予測グラフの「予測レンジ（±1円）」の幅や、急激な価格の跳ね上がり（スパイク）の兆候を併せてご確認いただき、本日の取引戦略の参考にしてください。
"""
            reports[area] = report_text
        
        progress_bar.progress(80)
        status_text.text("5/5: バックテスト評価とUIをレンダリング中...")

        # --- 結果表示 ---
        day_type_str = "休日(土日祝)" if target_is_dayoff else "平日"
        st.header(f"📅 {predict_date.strftime('%Y年%m月%d日')} ({day_type_str}) の予測・バックテスト結果")
        
        res_df = pd.DataFrame(predictions, index=target_features.index)
        
        actual_available = imb_df[(imb_df.index >= predict_dt) & (imb_df.index < predict_dt + timedelta(days=1))]
        is_backtest = not actual_available.empty and len(actual_available) > 0

        tabs = st.tabs(AREAS)
        for i, area in enumerate(AREAS):
            with tabs[i]:
                st.info(reports[area])

                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=res_df.index, y=res_df[area] + 1.0,
                    mode='lines', line=dict(width=0), showlegend=False, name='Upper'
                ))
                fig.add_trace(go.Scatter(
                    x=res_df.index, y=res_df[area] - 1.0,
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(0,176,246,0.2)',
                    showlegend=True, name='予測レンジ (±1円)'
                ))
                
                fig.add_trace(go.Scatter(
                    x=res_df.index, y=res_df[area],
                    mode='lines+markers', name='予測価格',
                    line=dict(color='blue', width=2)
                ))

                spot_col_name = f"{area}_スポット"
                if spot_col_name in target_features.columns:
                    fig.add_trace(go.Scatter(
                        x=target_features.index, y=target_features[spot_col_name],
                        mode='lines', name='スポット価格',
                        line=dict(color='green', dash='dashdot', width=2)
                    ))

                if is_backtest and area in actual_available.columns:
                    actual = actual_available[area]
                    fig.add_trace(go.Scatter(
                        x=actual.index, y=actual,
                        mode='lines', name='実績価格',
                        line=dict(color='red', dash='dot', width=2)
                    ))
                    
                    compare_df = pd.concat([res_df[area].rename('Pred'), actual.rename('Actual')], axis=1).dropna()
                    if not compare_df.empty:
                        mae = np.mean(np.abs(compare_df['Pred'] - compare_df['Actual']))
                        
                        compare_df['Within_Range'] = (compare_df['Actual'] >= compare_df['Pred'] - 1.0) & (compare_df['Actual'] <= compare_df['Pred'] + 1.0)
                        hit_count = compare_df['Within_Range'].sum()
                        total_count = len(compare_df)
                        hit_rate = (hit_count / total_count) * 100
                        
                        col1, col2 = st.columns(2)
                        col1.metric("平均絶対誤差 (MAE)", f"{mae:.2f} 円")
                        col2.metric("レンジ的中率 (±1円以内)", f"{hit_count}/{total_count} コマ", f"{hit_rate:.1f}%")

                fig.update_layout(
                    title=f"{area}エリア 30分コマ インバランス料金推移",
                    xaxis_title="時刻",
                    yaxis_title="価格 (円/kWh)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 予測データ一覧 (CSVダウンロード可能)")
        st.dataframe(res_df.style.format("{:.2f}"))
        
        progress_bar.progress(100)
        status_text.text("全処理が完了しました。")