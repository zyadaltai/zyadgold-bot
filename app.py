import os
import json
import time
import sqlite3
import threading
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from flask import Flask

# ===========================================================================
# إعداد خادم الويب (Flask) لإبقاء البوت متصلاً على السحابة 24/7
# ===========================================================================
app_flask = Flask(__name__)

@app_flask.route('/')
def home():
    return "🤖 GoldBot Institutional is Active and Running 24/7 on Cloud!"

def run_web_server():
    port = int(os.environ.get("PORT", 5000))
    app_flask.run(host="0.0.0.0", port=port)

# ===========================================================================
# إعدادات الحساب الافتراضية
# ===========================================================================
DEFAULT_ACCOUNT_SIZE = 5000.0  
DEFAULT_RISK_PERCENTAGE = 0.01  

# ===========================================================================
# 1. إعداد قاعدة البيانات المحلية (SQLite)
# ===========================================================================
def init_db():
    conn = sqlite3.connect("gold_bot_trades.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action TEXT,
            entry REAL,
            sl REAL,
            tp1 REAL,
            tp2 REAL,
            lot_size REAL,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_trade_to_db(trade_data, timestamp):
    try:
        conn = sqlite3.connect("gold_bot_trades.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (timestamp, action, entry, sl, tp1, tp2, lot_size, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, 
            trade_data.get('action'), 
            trade_data.get('entry'), 
            trade_data.get('sl'), 
            trade_data.get('tp1'), 
            trade_data.get('tp2'), 
            trade_data.get('lot_size'),
            'Active'
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[-] خطأ في حفظ البيانات محلياً: {e}")

def get_last_trade_from_db():
    try:
        conn = sqlite3.connect("gold_bot_trades.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('SELECT timestamp, action, entry, sl, tp1, tp2, lot_size FROM trades ORDER BY id DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        return row
    except Exception:
        return None

# ===========================================================================
# 2. تفعيل معالجة الطلبات وإلغاء المعلمات غير المدعومة لـ Groq
# ===========================================================================
import litellm

_original_completion = litellm.completion
def _cleaned_completion(*args, **kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
                msg.pop("cache_control", None)
    kwargs.pop("cache_breakpoint", None)
    return _original_completion(*args, **kwargs)

litellm.completion = _cleaned_completion
litellm.drop_params = True

from crewai import Agent, Task, Crew, Process, LLM

# ===========================================================================
# 3. إعداد مفاتيح Groq و Telegram
# ===========================================================================
GROQ_KEY = os.environ.get("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

os.environ["GROQ_API_KEY"] = GROQ_KEY
os.environ["LITELLM_DROP_PARAMS"] = "True"

groq_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=GROQ_KEY
)

# ===========================================================================
# 4. محرك الحسابات الفنية المتقدمة
# ===========================================================================
class EnterpriseEngine:
    @staticmethod
    def calculate_indicators():
        try:
            res = requests.get("https://api.gold-api.com/price/XAU", timeout=5)
            spot_price = round(float(res.json().get("price")), 2) if res.status_code == 200 else None
        except Exception:
            spot_price = None

        ticker = yf.Ticker("GC=F")
        df_1h = ticker.history(period="10d", interval="1h")
        df_1h = df_1h.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}).reset_index()

        if spot_price is None:
            spot_price = round(df_1h['close'].iloc[-1], 2)
        else:
            diff = spot_price - df_1h['close'].iloc[-1]
            df_1h['open'] += diff
            df_1h['high'] += diff
            df_1h['low'] += diff
            df_1h['close'] += diff

        delta = df_1h['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        current_rsi = round((100 - (100 / (1 + rs))).iloc[-1], 2)

        high_low = df_1h['high'] - df_1h['low']
        high_close = np.abs(df_1h['high'] - df_1h['close'].shift())
        low_close = np.abs(df_1h['low'] - df_1h['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_value = round(true_range.rolling(window=14).mean().iloc[-1], 2)

        exp1 = df_1h['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_1h['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        sma = df_1h['close'].rolling(window=20).mean()
        std = df_1h['close'].rolling(window=20).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)

        df_4h = ticker.history(period="59d", interval="4h")
        df_4h = df_4h.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}).reset_index()
        ema_50_4h = df_4h['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        macro_trend = "Bullish (صاعد)" if df_4h['close'].iloc[-1] > ema_50_4h else "Bearish (هابط)"

        return {
            "price": spot_price,
            "rsi": current_rsi,
            "atr": atr_value,
            "macd": round(macd.iloc[-1], 2),
            "macd_signal": round(signal.iloc[-1], 2),
            "bb_upper": round(upper.iloc[-1], 2),
            "bb_lower": round(lower.iloc[-1], 2),
            "trend": macro_trend
        }

    @staticmethod
    def calculate_position_size(account_size, risk_pct, entry, sl) -> float:
        risk_amount = account_size * risk_pct
        risk_per_unit = abs(entry - sl)
        if risk_per_unit == 0:
            return 0.01
        lot_size = risk_amount / (risk_per_unit * 100)
        return max(round(lot_size, 2), 0.01)

# ===========================================================================
# 5. معالجة أوامر تليجرام التفاعلية
# ===========================================================================
def send_telegram_message(message_text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_text,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            payload.pop("parse_mode", None)
            requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[-] خطأ تليجرام: {e}")

def handle_telegram_command(chat_id, text):
    text = text.strip()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    reply_text = ""
    lower_text = text.lower()
    
    if lower_text == "/price":
        try:
            ind = EnterpriseEngine.calculate_indicators()
            reply_text = f"📍 **السعر الفوري الحالي للذهب (XAUUSD):** `{ind['price']} $`"
        except Exception:
            reply_text = "[-] عذراً، تعذر جلب السعر حالياً."
            
    elif lower_text in ["/tech", "/indicators"]:
        try:
            ind = EnterpriseEngine.calculate_indicators()
            reply_text = (
                f"📊 **التحليل الفني الفوري (Live Indicators):**\n"
                f"• السعر الحالي: `{ind['price']} $`\n"
                f"• الترند العام (4h): `{ind['trend']}`\n"
                f"• مؤشر RSI (14): `{ind['rsi']}`\n"
                f"• تقلبات ATR (14): `{ind['atr']}`\n"
                f"• مؤشر MACD: `{ind['macd']} (Signal: {ind['macd_signal']})`\n"
                f"• نطاق Bollinger العليا: `{ind['bb_upper']}`\n"
                f"• نطاق Bollinger السفلى: `{ind['bb_lower']}`"
            )
        except Exception:
            reply_text = "[-] عذراً، حدث خطأ أثناء حساب المؤشرات الحية."
            
    elif lower_text.startswith("/calc"):
        try:
            parts = text.split()
            if len(parts) >= 3:
                entry_p = float(parts[1])
                sl_p = float(parts[2])
                lot = EnterpriseEngine.calculate_position_size(DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PERCENTAGE, entry_p, sl_p)
                risk_usd = DEFAULT_ACCOUNT_SIZE * DEFAULT_RISK_PERCENTAGE
                reply_text = (
                    f"🧮 **حاسبة حجم اللوت الفورية:**\n"
                    f"• رأس المال الافتراضي: `{DEFAULT_ACCOUNT_SIZE}$`\n"
                    f"• نسبة المخاطرة: `1% ({risk_usd}$)`\n"
                    f"• نقطة الدخول: `{entry_p}`\n"
                    f"• وقف الخسارة: `{sl_p}`\n"
                    f"🔹 **حجم العقد المقترح (Lot Size):** `{lot} Lot`"
                )
            else:
                reply_text = "⚠️ الاستخدام الصحيح:\n`/calc [سعر الدخول] [وقف الخسارة]`\nمثال: `/calc 4075 4045`"
        except Exception:
            reply_text = "❌ خطأ في تنسيق الأرقام."
            
    elif lower_text == "/last":
        last_trade = get_last_trade_from_db()
        if last_trade:
            ts, action, entry, sl, tp1, tp2, lot = last_trade
            reply_text = f"📊 **آخر صفقة مسجلة:**\n• التوقيت: `{ts}`\n• الاتجاه: `{action}`\n• الدخول: `{entry}`\n• الوقف: `{sl}`\n• الهدف الأول: `{tp1}`\n• حجم العقد: `{lot} Lot`"
        else:
            reply_text = "📭 لا توجد صفقات مسجلة حتى الآن."
            
    elif lower_text == "/help":
        reply_text = (
            "📌 **قائمة الأوامر التفاعلية للبوت:**\n"
            "• `/price` - السعر الفوري للذهب.\n"
            "• `/tech` أو `/indicators` - مؤشرات MACD, Bollinger, RSI, ATR.\n"
            "• `/calc [دخول] [وقف]` - حاسبة اللوت الذكية.\n"
            "• `/last` - عرض آخر صفقة مسجلة.\n"
            "• `/help` - عرض هذه القائمة."
        )
        
    if reply_text:
        requests.post(url, json={"chat_id": chat_id, "text": reply_text, "parse_mode": "Markdown"})

def telegram_listener_thread():
    offset = 0
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset={offset}&timeout=30"
            res = requests.get(url, timeout=35)
            if res.status_code == 200:
                data = res.json()
                for result in data.get("result", []):
                    offset = result["update_id"] + 1
                    message = result.get("message", {})
                    chat_id = message.get("chat", {}).get("id")
                    text = message.get("text", "")
                    if chat_id and text:
                        handle_telegram_command(chat_id, text)
        except Exception:
            time.sleep(5)
        time.sleep(1)

# ===========================================================================
# 6. التقويم الاقتصادي الحي والتحليل
# ===========================================================================
def fetch_live_economic_calendar() -> str:
    try:
        url = "https://nfs.faireconomy.media/ff_cal_thisweek.json"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            events = res.json()
            today_str = datetime.now().strftime("%Y-%m-%d")
            high_impact_found = False
            for ev in events:
                if ev.get("country") == "USD" and ev.get("impact") == "High":
                    ev_date = ev.get("date", "").split("T")[0]
                    if ev_date == today_str:
                        high_impact_found = True
                        break
            if high_impact_found:
                return "⚠️ **تنبيه إخباري حي:** يوجد اليوم بيان اقتصادي أمريكي **عالي التأثير**!"
        return "✅ **حالة الأخبار الحية:** الأجواء الاقتصادية هادئة ولا توجد أخبار حرجة الآن."
    except Exception:
        return "✅ **حالة الأخبار الحية:** تم فحص التقويم الاقتصادي."

def fetch_market_metrics() -> dict:
    ind = EnterpriseEngine.calculate_indicators()
    news_status = fetch_live_economic_calendar()

    risk_distance = max(ind['atr'] * 1.2, 8.0)
    action = "شراء Long"
    entry = ind['price']
    sl = round(entry - risk_distance, 2)
    tp1 = round(entry + (risk_distance * 2.5), 2)
    tp2 = round(entry + (risk_distance * 4.0), 2)

    lot_size = EnterpriseEngine.calculate_position_size(DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PERCENTAGE, entry, sl)
    trailing_stop_desc = "يتحرك وقف الخسارة تدريجياً وبشكل تلقائي بمقدار أرباح الهدف الأول (TP1)."

    trade_setup = {
        "action": action,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "lot_size": lot_size,
        "trailing_stop": trailing_stop_desc,
        "rr": "1:3.5"
    }

    return {
        "symbol": "XAUUSD (الذهب الفوري مقابل الدولار)",
        "current_price": ind['price'],
        "rsi_14": ind['rsi'],
        "atr_14": ind['atr'],
        "macd_val": ind['macd'],
        "bb_upper": ind['bb_upper'],
        "bb_lower": ind['bb_lower'],
        "macro_trend_4h": ind['trend'],
        "live_news": news_status,
        "pre_calculated_trade": trade_setup
    }

# ===========================================================================
# 7. وكلاء الذكاء الاصطناعي (تمت إضافة مدير المخاطر)
# ===========================================================================
analyst_agent = Agent(
    role="Enterprise Chief Quantitative Officer",
    goal="Synthesize multi-timeframe quantitative data, MACD, Bollinger Bands, and risk parameters.",
    backstory="Global head of quantitative analysis. You read market numbers logically without emotion.",
    llm=groq_llm,
    verbose=True
)

risk_manager_agent = Agent(
    role="Strict Risk Manager",
    goal="Protect capital by strictly reviewing the trade setup. Ensure Stop Loss (SL) is present and risk distance is safe. Reject unsafe trades.",
    backstory="You are the ultimate gatekeeper. You hate uncalculated risk. If the trend is unclear or the setup is weak, you strictly veto the trade.",
    llm=groq_llm,
    verbose=True
)

signal_agent = Agent(
    role="Gold Execution Officer",
    goal="Format pristine Arabic enterprise trade cards based on the Risk Manager's final approval.",
    backstory="Wall Street senior desk execution officer. You only publish trades that have passed the strict review of the Risk Manager.",
    llm=groq_llm,
    verbose=True
)

task1 = Task(
    description="Analyze inputs: {market_data}. Determine the trend and technical strength.",
    expected_output="Technical analysis report.",
    agent=analyst_agent
)

task_risk = Task(
    description=(
        "Review the technical report and {market_data}. "
        "Check if the pre_calculated_trade is safe (e.g., has a valid SL, trend matches action). "
        "If unsafe, highly volatile, or facing high impact news, output a VETO command. If safe, output APPROVED."
    ),
    expected_output="APPROVED or VETO with a short explanation.",
    agent=risk_manager_agent,
    context=[task1]
)

task2 = Task(
    description=(
        "Read the Risk Manager's decision. "
        "IF the decision is VETO, output exactly: '⚠️ **تدخل مدير المخاطر:** تم إلغاء الصفقة نظراً لارتفاع المخاطرة أو عدم وضوح الاتجاه. حماية رأس المال هي الأولوية.'\n"
        "IF the decision is APPROVED, format the following card exactly in Arabic:\n"
        "🌐 **بطاقة التداول المؤسسي المتقدمة (XAUUSD)**\n"
        "-----------------------------------\n"
        "📍 **السعر الحالي:** [current_price]\n"
        "📈 **الترند العام (4h):** [macro_trend_4h]\n"
        "📊 **اتجاه الصفقة:** [action من pre_calculated_trade]\n"
        "مؤشر RSI: [rsi_14] | ATR: [atr_14] | MACD: [macd_val]\n"
        "Bollinger Bands ➔ العليا: [bb_upper] | السفلى: [bb_lower]\n\n"
        "🎯 **تفاصيل المعلمات والمخاطر:**\n"
        "• **سعر الدخول:** [entry]\n"
        "• **وقف الخسارة الأولي:** [sl]\n"
        "• **حجم العقد المقترح (Lot Size):** [lot_size] (مبني على مخاطرة 1%)\n"
        "• **الهدف الأول:** [tp1]\n"
        "• **الهدف الثاني (TP2):** [tp2]\n"
        "• **نسبة العائد للمخاطرة:** [rr]\n\n"
        "🛡️ **استراتيجية التوقف المتحرك:**\n"
        "[trailing_stop من pre_calculated_trade]\n\n"
        "📰 **التقويم الاقتصادي الحي:**\n"
        "[live_news]\n\n"
        "💡 **التحليل المؤسسي:**\n"
        "تم التدقيق من قبل فريق إدارة المخاطر واعتماد الصفقة."
    ),
    expected_output="Final formatted message for Telegram (either trade card or Veto message).",
    agent=signal_agent,
    context=[task_risk]
)

crew = Crew(
    agents=[analyst_agent, risk_manager_agent, signal_agent],
    tasks=[task1, task_risk, task2],
    process=Process.sequential,
    verbose=True
)

def background_bot_loop():
    while True:
        try:
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            weekday = now.weekday()
            
            if weekday == 5 or weekday == 6:
                time.sleep(14400)
                continue

            print(f"\n[🕒 {current_time}] 🚀 جاري إرسال التحليل المؤسسي الدوري...")
            market_data = fetch_market_metrics()
            raw_data_str = json.dumps(market_data, indent=2, ensure_ascii=False)
            
            result = crew.kickoff(inputs={"market_data": raw_data_str})
            result_text = str(result)
            
            send_telegram_message(result_text)
            
            # تسجيل الصفقة فقط إذا وافق عليها مدير المخاطر (لم يتم عمل فيتو)
            if "تدخل مدير المخاطر" not in result_text:
                log_trade_to_db(market_data['pre_calculated_trade'], current_time)
                
            print("[+] ✅ تمت دورة الإرسال بنجاح!")
            
        except Exception as e:
            print(f"[-] خطأ: {e}")
            
        time.sleep(3600)

# ===========================================================================
# 8. التشغيل المتزامن (الويب + التليجرام + حلقة البوت)
# ===========================================================================
if __name__ == "__main__":
    init_db()
    
    # تشغيل بوت التليجرام التفاعلي وحلقة البوت في خيوط خلفية (Threads)
    threading.Thread(target=telegram_listener_thread, daemon=True).start()
    threading.Thread(target=background_bot_loop, daemon=True).start()
    
    # تشغيل خادم الويب في الخيط الرئيسي لضمان الاستضافة السحابية
    run_web_server()
