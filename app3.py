# app_dashboard.py
# --- 終極儀表板 - v3.1 ---

import sqlite3
import pandas as pd
from flask import Flask, jsonify, render_template_string, url_for, abort, make_response

# --- 1. 初始化與設定 ---
app = Flask(__name__)
DATABASE = 'yydata_Tyy.db'


# --- 2. 後端資料庫函式 ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE, timeout=10)
    return conn


def get_full_event_list():
    """獲取完整的事件列表，用於日誌顯示。最新的在最上面。"""
    try:
        with get_db_connection() as conn:
            query = "SELECT * FROM fight ORDER BY Book_ID DESC;"
            df = pd.read_sql_query(query, conn)
        return df
    except sqlite3.OperationalError as e:
        print(f"Database list read error: {e}")
        return pd.DataFrame()


def get_latest_event():
    """精準地只獲取 ID 最大的那一筆紀錄（忽略初始資料），用於儀表板。"""
    try:
        with get_db_connection() as conn:
            query = "SELECT * FROM fight WHERE Book_ID != 1000 ORDER BY Book_ID DESC LIMIT 1;"
            df = pd.read_sql_query(query, conn)
        return df
    except sqlite3.OperationalError as e:
        print(f"Database latest event read error: {e}")
        return pd.DataFrame()


def get_single_record(record_id):
    """根據 ID 獲取單一筆紀錄，用於詳情頁。"""
    try:
        with get_db_connection() as conn:
            query = f"SELECT * FROM fight WHERE Book_ID = ?;"
            df = pd.read_sql_query(query, conn, params=(record_id,))
        return df
    except sqlite3.OperationalError as e:
        print(f"Database single record read error: {e}")
        return pd.DataFrame()


# --- 3. API 與頁面路由 ---
@app.route('/api/data')
def api_data():
    latest_event_df = get_latest_event()
    event_list_df = get_full_event_list()

    latest_event_dict = None
    if not latest_event_df.empty:
        latest_event_dict = latest_event_df.iloc[0].to_dict()
        for k, v in latest_event_dict.items():
            if hasattr(v, 'item'): latest_event_dict[k] = v.item()

    event_list = []
    if not event_list_df.empty:
        event_list = event_list_df.to_dict(orient='records')
        for event in event_list:
            for k, v in event.items():
                if hasattr(v, 'item'): event[k] = v.item()

    response = make_response(jsonify(latest_event=latest_event_dict, event_list=event_list))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


@app.route('/')
def index():
    return render_template_string(DASHBOARD_TEMPLATE)


@app.route('/details/<int:record_id>')
def details(record_id):
    record_df = get_single_record(record_id)
    if record_df.empty: abort(404)
    details_dict = record_df.iloc[0].to_dict()
    for k, v in details_dict.items():
        if hasattr(v, 'item'): details_dict[k] = v.item()
    # 這裡會自動生成圖片路徑，例如 /static/1.jpg
    details_dict['image_url'] = url_for('static', filename=f"{details_dict['Book_ID']}.jpg")
    return render_template_string(DETAIL_PAGE_TEMPLATE, details=details_dict)


# --- 4. HTML 模板 ---
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <title>ATAS 威脅感知系統 v3.1</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
        :root { --primary-color: #00e5ff; --bg-color: #020a1a; --panel-color: rgba(10, 25, 47, 0.8); --border-color: rgba(0, 229, 255, 0.3); --text-color: #c9d1d9; --text-muted: #8b949e; --danger-color: #ff4757; --safe-color: #2ed573; --warn-color: #f9c859; }
        body { font-family: 'Microsoft JhengHei', sans-serif; background-color: var(--bg-color); color: var(--text-color); margin: 0; padding: 20px; overflow: hidden; background-image: radial-gradient(var(--border-color) 1px, transparent 1px); background-size: 30px 30px; background-position: 0 0, 15px 15px; }
        #loader { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: var(--bg-color); z-index: 9999; display: flex; flex-direction: column; justify-content: center; align-items: center; font-family: 'Orbitron', sans-serif; transition: opacity 0.8s ease-out; }
        #loader .loading-text { font-size: 2em; color: var(--primary-color); text-shadow: 0 0 10px var(--primary-color); }
        .progress-bar { width: 300px; height: 3px; background: var(--border-color); margin-top: 20px; }
        .progress-bar div { width: 0; height: 100%; background: var(--primary-color); animation: loading 2.5s ease-out forwards; }
        .dashboard { display: grid; grid-template-columns: 1fr 1.8fr 1.2fr; gap: 20px; height: calc(100vh - 40px); visibility: hidden; opacity: 0; transition: opacity 0.5s 0.3s; }
        .left-column, .right-column { display: flex; flex-direction: column; gap: 20px; min-height: 0; }
        .panel { background: var(--panel-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; backdrop-filter: blur(5px); display: flex; flex-direction: column; }
        .panel-title { color: var(--primary-color); font-size: 1.5em; margin: 0 0 15px 0; padding-bottom: 10px; border-bottom: 1px solid var(--border-color); text-align: center; }
        .center-panel { align-items: center; justify-content: center; }
        .center-visual { display: flex; justify-content: center; align-items: center; width: 100%; height: 100%;}
        .center-visual img { max-width: 90%; max-height: 90%; height: auto; opacity: 0.7; filter: drop-shadow(0 0 15px var(--primary-color)); }
        .chart-container { flex-grow: 1; min-height: 200px; }
        .left-column .panel { flex: 1; }
        .right-column .panel:first-child { flex: 2; }
        #event-log-panel { flex: 3; overflow: hidden; }
        #event-log { flex-grow: 1; overflow-y: auto; }
        .log-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 5px; border-bottom: 1px solid #30363d; cursor: pointer; transition: background-color 0.2s; }
        .log-item:hover { background-color: rgba(88, 166, 255, 0.1); }
        .log-id { color: var(--text-muted); margin-right: 10px; }
        .log-total.danger { color: var(--danger-color); }
        .log-total.warn { color: var(--warn-color); }
        .log-total.safe { color: var(--safe-color); }
        @keyframes loading { from { width: 0; } to { width: 100%; } }
    </style>
</head>
<body>
    <div id="loader"><div class="loading-text">系統開啟中...</div><div class="progress-bar"><div></div></div></div>
    <div id="main-dashboard" class="dashboard">
        <div class="left-column">
            <div class="panel"><h2 class="panel-title">綜合狀態評估</h2><div id="radar-chart" class="chart-container"></div></div>
            <div class="panel"><h2 class="panel-title">距離 (m)</h2><div id="distance-gauge" class="chart-container"></div></div>
        </div>
        <div class="panel center-panel">
            <div class="center-visual"><img src="/static/body_scan.gif" alt="Body Scan"></div>
        </div>
        <div class="right-column">
            <div class="panel"><h2 class="panel-title">危險總值</h2><div id="total-gauge" class="chart-container"></div></div>
            <div class="panel" id="event-log-panel"><h2 class="panel-title">事件日誌</h2><div id="event-log"></div></div>
        </div>
    </div>
    <script>
        window.addEventListener('load', () => {
            const loader = document.getElementById('loader');
            const dashboard = document.getElementById('main-dashboard');
            setTimeout(() => {
                loader.style.opacity = '0';
                dashboard.style.visibility = 'visible';
                dashboard.style.opacity = '1';
                setTimeout(() => { loader.style.display = 'none'; }, 500);
            }, 2500);
        });
        const radarChart = echarts.init(document.getElementById('radar-chart'), 'dark');
        const distanceGauge = echarts.init(document.getElementById('distance-gauge'), 'dark');
        const totalGauge = echarts.init(document.getElementById('total-gauge'), 'dark');
        const eventLog = document.getElementById('event-log');
        async function updateDashboard() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                if (!data || !data.latest_event) {
                    // 如果沒有最新數據，可以清空或顯示預設畫面
                    return;
                }
                const latest = data.latest_event;
                const poseScore = latest.Pose.toLowerCase().includes('stand') ? 20 : 80;
                const emoScore = ['Sad', 'Fear', 'Disgust', 'Angry'].includes(latest.Face) ? 90 : (latest.Face === 'Happy' ? 10 : 40);
                const yoloScore = latest.Yolo === 'people' ? 10 : 95;
                const distScore = latest.Distance < 1.5 ? 85 : 25;
                radarChart.setOption({ radar: { indicator: [ { name: '姿勢異常', max: 100 }, { name: '負面情緒', max: 100 }, { name: '物件威脅', max: 100 }, { name: '距離過近', max: 100 } ], shape: 'circle', center: ['50%', '55%']}, series: [{ type: 'radar', data: [{ value: [poseScore, emoScore, yoloScore, distScore] }]}] });
                distanceGauge.setOption({ series: [{ type: 'gauge', min: 0, max: 5, splitNumber: 5, progress: { show: true }, detail: { valueAnimation: true, formatter: '{value}m', fontSize: 20, offsetCenter: [0, '70%'] }, data: [{ value: latest.Distance.toFixed(2) }] }] });
                totalGauge.setOption({ series: [{ type: 'gauge', min: 0, max: 5000, splitNumber: 5, axisLine: { lineStyle: { width: 20, color: [[0.65, '#3fb950'], [0.85, '#f9c859'], [1, '#ff4757']] } }, progress: { show: true, width: 20 }, detail: { valueAnimation: true, fontSize: 30, offsetCenter: [0, '60%'] }, data: [{ value: latest.Total }] }] });

                eventLog.innerHTML = '';
                const display_list = data.event_list;
                display_list.forEach(event => {
                    const item = document.createElement('div');
                    item.className = 'log-item';
                    item.onclick = () => { window.location.href = `/details/${event.Book_ID}`; };
                    let totalClass = 'safe';
                    if (event.Total > 4250) totalClass = 'danger';
                    else if (event.Total > 3275) totalClass = 'warn';
                    item.innerHTML = `<span class="log-id">#${event.Book_ID}</span><span class="log-yolo">${event.Yolo}</span><span class="log-total ${totalClass}">${event.Total}</span>`;
                    eventLog.appendChild(item);
                });
            } catch (e) { console.error("Update failed:", e); }
        }
        setTimeout(() => {
            updateDashboard();
            setInterval(updateDashboard, 1500);
        }, 3000);
    </script>
</body>
</html>
"""
DETAIL_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <title>事件詳情 #{{ details.Book_ID }}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
        body { font-family: 'Microsoft JhengHei', sans-serif; background-color: #0d1117; color: #c9d1d9; padding: 40px; }
        .container { max-width: 1000px; margin: auto; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 40px; }
        .header { color: #58a6ff; font-family: 'Orbitron', sans-serif; }
        .header-status.danger { color: #f85149; } .header-status.safe { color: #3fb950; }
        .content { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 30px; margin-top: 20px; align-items: start;}
        img { width: 100%; border-radius: 6px; border: 1px solid #30363d; }
        dl { display: grid; grid-template-columns: auto 1fr; gap: 12px 20px; }
        dt { color: #8b949e; } dd { margin: 0; font-size: 1.1em; word-break: break-word; }
        dd.danger { color: #f85149; font-weight: bold; }
        a { color: #58a6ff; text-decoration: none; display: inline-block; margin-top: 30px; font-size: 1.1em; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="header">
            EVENT #{{ details.Book_ID }} ANALYSIS
            <span class="{{ 'danger' if details.Total > 3275 else 'safe' }}">
                ({% if details.Total > 3275 %}High{% else %}Normal{% endif %} Threat Level)
            </span>
        </h1>
        <div class="content">
            <div class="image-container"><img src="{{ details.image_url }}" alt="Event Snapshot"></div>
            <div class="details-container">
                <dl>
                    <dt>類別 (YOLO):</dt><dd>{{ details.Yolo }}</dd>
                    <dt>表情 (Face):</dt><dd>{{ details.Face }}</dd>
                    <dt>姿勢 (Pose):</dt><dd>{{ details.Pose }}</dd>
                    <dt>年齡 (Age):</dt><dd>{{ details.Age }}</dd>
                    <dt>性別 (Gender):</dt><dd>{{ details.Gender }}</dd>
                    <dt>距離 (Distance):</dt><dd>{{ "%.2f"|format(details.Distance) }} m</dd>
                    <dt>危險值 (Total):</dt><dd class="{{ 'danger' if details.Total > 3275 else '' }}">{{ details.Total }}</dd>
                </dl>
               <a href="javascript:history.back()">← Back to Dashboard</a>
            </div>
        </div>
    </div>
</body>
</html>
"""

# --- 5. 啟動伺服器 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)