import os
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ─── AYARLAR ──────────────────────────────────────────────────────────────────

TAKIP_LISTESI = [
    "THYAO", "ASELS", "BIMAS", "FROTO", "SASA",
    "EREGL", "KCHOL", "GARAN", "AKBNK", "TUPRS",
    "PGSUS", "TAVHL", "TCELL", "TTKOM", "MGROS",
    "ARCLK", "VESTL", "KOZAL", "EKGYO", "TOASO",
]

TREND_EVREN = [
    "THYAO", "ASELS", "BIMAS", "FROTO", "SASA", "EREGL", "KCHOL",
    "GARAN", "AKBNK", "TUPRS", "PGSUS", "TAVHL", "TCELL", "TTKOM",
    "MGROS", "ARCLK", "VESTL", "KOZAL", "EKGYO", "TOASO", "PETKM",
]

# ─── TEKNİK ANALİZ FONKSİYONLARI ──────────────────────────────────────────────

def rsi_hesapla(fiyatlar, periyot=14):
    delta = fiyatlar.diff()
    kazanc = delta.where(delta > 0, 0).rolling(periyot).mean()
    kayip = (-delta.where(delta < 0, 0)).rolling(periyot).mean()
    if kayip.iloc[-1] == 0: return 100.0
    rs = kazanc.iloc[-1] / kayip.iloc[-1]
    return round(100 - (100 / (1 + rs)), 1)

def macd_hesapla(fiyatlar):
    ema12 = fiyatlar.ewm(span=12, adjust=False).mean()
    ema26 = fiyatlar.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sinyal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sinyal
    return round(macd.iloc[-1], 3), round(sinyal.iloc[-1], 3), round(hist.iloc[-1], 3)

def bollinger_hesapla(fiyatlar, periyot=20):
    ma = fiyatlar.rolling(periyot).mean()
    std = fiyatlar.rolling(periyot).std()
    ust = ma + std * 2
    alt = ma - std * 2
    son = fiyatlar.iloc[-1]
    poz = round(((son - alt.iloc[-1]) / (ust.iloc[-1] - alt.iloc[-1])) * 100, 1) if (ust.iloc[-1] != alt.iloc[-1]) else 50.0
    return round(ust.iloc[-1], 2), round(ma.iloc[-1], 2), round(alt.iloc[-1], 2), poz

def sinyal_uret(rsi, macd_h, degisim, hacim_r, bb_p, ma20, ma50, fiyat):
    skor = 0
    sinyaller = []
    if rsi < 35: skor += 2; sinyaller.append("RSI Satım Bölgesi")
    elif rsi > 65: skor -= 2; sinyaller.append("RSI Alım Bölgesi")
    if macd_h > 0: skor += 1; sinyaller.append("MACD Pozitif")
    if fiyat > ma20: skor += 1
    
    if skor >= 2: karar = "AL"
    elif skor <= -2: karar = "SAT"
    else: karar = "TUT"
    return karar, sinyaller

# ─── VERİ ÇEKME VE ANALİZ ─────────────────────────────────────────────────────

def hisse_analiz(ticker):
    try:
        df = yf.Ticker(ticker + ".IS").history(period="6mo")
        if df.empty or len(df) < 30: return None
        fiyatlar = df["Close"]
        son = round(fiyatlar.iloc[-1], 2)
        degisim = round(((son - fiyatlar.iloc[-2]) / fiyatlar.iloc[-2]) * 100, 2)
        
        rsi = rsi_hesapla(fiyatlar)
        macd, _, macd_h = macd_hesapla(fiyatlar)
        _, ma20, _, bb_p = bollinger_hesapla(fiyatlar)
        ma50 = fiyatlar.rolling(50).mean().iloc[-1]
        hacim_r = round(df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1], 2)

        karar, sinyaller = sinyal_uret(rsi, macd_h, degisim, hacim_r, bb_p, ma20, ma50, son)
        return {"ticker": ticker, "fiyat": son, "degisim": degisim, "rsi": rsi, "macd_hist": macd_h, "hacim_orani": hacim_r, "bb_poz": bb_p, "karar": karar, "sinyaller": sinyaller}
    except: return None

def doviz_cek():
    try:
        r = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5).json()
        return {"USD/TRY": round(r["rates"]["TRY"], 2), "EUR/TRY": round(r["rates"]["TRY"] / r["rates"]["EUR"], 2)}
    except: return {"USD/TRY": "0.0", "EUR/TRY": "0.0"}

# ─── GEMINI AI ENTEGRASYONU (DÜZENLENDİ) ──────────────────────────────────────

def gemini_istek(prompt, max_tokens=1000):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key: return "HATA: API Anahtarı eksik."

    # Model ismi 1.5 Flash olarak güncellendi (Kota dostu)
 url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"  
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.5}
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        data = response.json()
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        return f"API Hatası: {data.get('error', {}).get('message', 'Bilinmeyen hata')}"
    except Exception as e:
        return f"Bağlantı Hatası: {str(e)}"

def ai_tavsiye_olustur(analizler, doviz):
    # Gemini'yi yormamak için sadece ilk 8 hisseyi gönderiyoruz
    veri_ozet = "\n".join([f"{a['ticker']}: {a['fiyat']}TL, RSI:{a['rsi']}, Karar:{a['karar']}" for a in analizler[:8]])
    prompt = f"Bir borsa analisti olarak şu verileri yorumla ve kısa tavsiyeler ver:\nKurlar: {doviz}\n\nHisseler:\n{veri_ozet}\n\nAnalizi Türkçe yap, kısa ve öz olsun."
    return gemini_istek(prompt)

# ─── TELEGRAM VE RAPORLAMA ───────────────────────────────────────────────────

def telegram_gonder(mesaj):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return print("Telegram bilgileri eksik.")
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": mesaj, "parse_mode": "HTML"})

def rapor_olustur(analizler, doviz, ai_yorum):
    tarih = datetime.now().strftime("%d.%m.%Y %H:%M")
    ozet = "\n".join([f"<b>{a['ticker']}</b>: {a['fiyat']} TL ({a['degisim']}%) -> {a['karar']}" for a in analizler[:10]])
    
    return f"""📊 <b>BIST GÜNLÜK RAPOR</b> ({tarih})
━━━━━━━━━━━━━━━━━━
💱 <b>PİYASA:</b> USD: {doviz['USD/TRY']} | EUR: {doviz['EUR/TRY']}

📋 <b>SİNYALLER:</b>
{ozet}

🤖 <b>AI YORUMU:</b>
{ai_yorum}

⚠️ <i>Yatırım tavsiyesi değildir.</i>"""

# ─── ANA ÇALIŞTIRICI ──────────────────────────────────────────────────────────

def main():
    print("🚀 Bot başlatıldı...")
    doviz = doviz_cek()
    
    analizler = []
    for ticker in TAKIP_LISTESI:
        res = hisse_analiz(ticker)
        if res: 
            analizler.append(res)
            print(f"✅ {ticker} analiz edildi.")
        time.sleep(0.2) # Yahoo Finance ban yememek için

    print("🤖 Gemini yorumu alınıyor...")
    ai_yorum = ai_tavsiye_olustur(analizler, doviz)
    
    rapor = rapor_olustur(analizler, doviz, ai_yorum)
    telegram_gonder(rapor)
    print("📤 Rapor gönderildi. İşlem tamam.")

if __name__ == "__main__":
    main()
