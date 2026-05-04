"""
BIST Günlük Borsa AI Ajan
- Yahoo Finance'dan gerçek BIST fiyatları
- Teknik analiz (RSI, MACD, Bollinger, MA)
- Gemini 1.5 Flash ile AI yorum (ücretsiz)
- Telegram'a günlük rapor
"""

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
    "ULKER", "LOGO", "MAVI", "HEKTS", "SOKM", "DOAS", "CIMSA",
    "OYAKC", "BRSAN", "SARKY", "KORDS", "NETAS", "ODAS", "VAKBN",
    "YKBNK", "ISCTR", "ENKAI", "CEMTS", "PARSN",
]

# ─── TEKNİK ANALİZ ────────────────────────────────────────────────────────────

def rsi_hesapla(fiyatlar, periyot=14):
    delta = fiyatlar.diff()
    kazanc = delta.where(delta > 0, 0).rolling(periyot).mean()
    kayip = (-delta.where(delta < 0, 0)).rolling(periyot).mean()
    if kayip.iloc[-1] == 0:
        return 100.0
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
    if ust.iloc[-1] == alt.iloc[-1]:
        poz = 50.0
    else:
        poz = round(((son - alt.iloc[-1]) / (ust.iloc[-1] - alt.iloc[-1])) * 100, 1)
    return round(ust.iloc[-1], 2), round(ma.iloc[-1], 2), round(alt.iloc[-1], 2), poz

def sinyal_uret(rsi, macd_hist, degisim, hacim_orani, bb_poz, ma20, ma50, fiyat):
    skor = 0
    sinyaller = []

    if rsi < 35:
        skor += 2
        sinyaller.append("RSI aşırı satım")
    elif rsi < 45:
        skor += 1
    elif rsi > 65:
        skor -= 2
        sinyaller.append("RSI aşırı alım")
    elif rsi > 55:
        skor -= 1

    if macd_hist > 0:
        skor += 1
        sinyaller.append("MACD pozitif")
    else:
        skor -= 1
        sinyaller.append("MACD negatif")

    if fiyat > ma20 > ma50:
        skor += 1
        sinyaller.append("MA yükselen trend")
    elif fiyat < ma20 < ma50:
        skor -= 1
        sinyaller.append("MA düşen trend")

    if hacim_orani > 1.5:
        skor += 1
        sinyaller.append(f"Hacim x{hacim_orani:.1f}")

    if bb_poz < 15:
        skor += 1
        sinyaller.append("Bollinger alt bant")
    elif bb_poz > 85:
        skor -= 1
        sinyaller.append("Bollinger üst bant")

    if degisim > 2:
        skor += 1
    elif degisim < -2:
        skor -= 1

    if skor >= 3:
        karar = "AL"
    elif skor <= -2:
        karar = "SAT"
    else:
        karar = "TUT"

    return karar, sinyaller, skor

def hisse_analiz(ticker):
    try:
        sembol = ticker + ".IS"
        hisse = yf.Ticker(sembol)
        df = hisse.history(period="6mo")

        if df is None or df.empty or len(df) < 30:
            return None

        fiyatlar = df["Close"]
        son = round(fiyatlar.iloc[-1], 2)
        onceki = round(fiyatlar.iloc[-2], 2)
        degisim = round(((son - onceki) / onceki) * 100, 2)

        ma20 = round(fiyatlar.rolling(20).mean().iloc[-1], 2)
        ma50 = round(fiyatlar.rolling(50).mean().iloc[-1], 2)
        rsi = rsi_hesapla(fiyatlar)
        macd, macd_s, macd_h = macd_hesapla(fiyatlar)
        bb_u, bb_m, bb_a, bb_p = bollinger_hesapla(fiyatlar)

        ort_hacim = df["Volume"].rolling(20).mean().iloc[-1]
        son_hacim = df["Volume"].iloc[-1]
        hacim_orani = round(son_hacim / ort_hacim, 2) if ort_hacim > 0 else 1.0

        son20 = df.tail(20)
        destek = round(son20["Low"].min(), 2)
        direnc = round(son20["High"].max(), 2)

        yillik = df.tail(252) if len(df) >= 252 else df
        yillik_yuksek = round(yillik["High"].max(), 2)
        yillik_dusuk = round(yillik["Low"].min(), 2)

        karar, sinyaller, skor = sinyal_uret(
            rsi, macd_h, degisim, hacim_orani, bb_p, ma20, ma50, son
        )

        return {
            "ticker": ticker,
            "fiyat": son,
            "degisim": degisim,
            "ma20": ma20,
            "ma50": ma50,
            "rsi": rsi,
            "macd_hist": macd_h,
            "bb_poz": bb_p,
            "bb_alt": bb_a,
            "bb_ust": bb_u,
            "hacim_orani": hacim_orani,
            "destek": destek,
            "direnc": direnc,
            "yillik_yuksek": yillik_yuksek,
            "yillik_dusuk": yillik_dusuk,
            "karar": karar,
            "sinyaller": sinyaller,
            "skor": skor,
        }
    except Exception as e:
        print(f"  {ticker} hata: {e}")
        return None

# ─── TREND TARAMA ─────────────────────────────────────────────────────────────

def trend_tara(min_hacim=1.5, top_n=8):
    print("Trend hisseler taranıyor...")
    sonuclar = []

    for ticker in TREND_EVREN:
        try:
            sembol = ticker + ".IS"
            df = yf.Ticker(sembol).history(period="3mo")
            if df is None or df.empty or len(df) < 25:
                continue

            son = df["Close"].iloc[-1]
            onceki = df["Close"].iloc[-2]
            gun_deg = ((son - onceki) / onceki) * 100

            hafta = df["Close"].tail(5)
            hafta_deg = ((hafta.iloc[-1] - hafta.iloc[0]) / hafta.iloc[0]) * 100

            ort_h = df["Volume"].rolling(20).mean().iloc[-1]
            son_h = df["Volume"].iloc[-1]
            hacim_r = son_h / ort_h if ort_h > 0 else 1.0

            if hacim_r >= min_hacim:
                momentum = abs(hafta_deg) * 0.4 + (hacim_r - 1) * 60 * 0.6
                sonuclar.append({
                    "ticker": ticker,
                    "fiyat": round(son, 2),
                    "gun_deg": round(gun_deg, 2),
                    "hafta_deg": round(hafta_deg, 2),
                    "hacim_r": round(hacim_r, 2),
                    "momentum": round(momentum, 1),
                })

            time.sleep(0.2)
        except Exception:
            continue

    sonuclar.sort(key=lambda x: x["momentum"], reverse=True)
    print(f"  {len(sonuclar)} trend hisse bulundu")
    return sonuclar[:top_n]

# ─── DÖVİZ ────────────────────────────────────────────────────────────────────

def doviz_cek():
    kurlar = {}
    try:
        r = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=8)
        d = r.json()
        if d.get("rates"):
            kurlar["USD/TRY"] = round(d["rates"].get("TRY", 0), 2)
            eur_usd = d["rates"].get("EUR", 1)
            kurlar["EUR/TRY"] = round(d["rates"].get("TRY", 0) / eur_usd, 2) if eur_usd else 0
    except Exception:
        pass

    try:
        r2 = requests.get("https://api.exchangerate-api.com/v4/latest/XAU", timeout=8)
        d2 = r2.json()
        if d2.get("rates"):
            gram_tl = d2["rates"].get("TRY", 0) / 31.1035
            kurlar["ALTIN_GRAM_TL"] = round(gram_tl, 0)
    except Exception:
        pass

    return kurlar

# ─── GEMİNİ ──────────────────────────────────────────────────────────────────

def gemini_istek(prompt, max_tokens=1500):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    
    print(f"  API Key yuklu mu: {bool(api_key)}")
    print(f"  API Key uzunlugu: {len(api_key)}")
    print(f"  API Key basi: {api_key[:8] if api_key else 'YOK'}")
    
    if not api_key:
        return "Gemini API anahtari eksik."

    # v1 ve v1beta, birden fazla model dene
    denemeler = [
        ("v1", "gemini-1.5-flash"),
        ("v1", "gemini-1.5-flash-001"),
        ("v1beta", "gemini-1.5-flash"),
        ("v1beta", "gemini-1.5-flash-latest"),
        ("v1beta", "gemini-2.0-flash"),
        ("v1", "gemini-1.0-pro"),
    ]

    for versiyon, model in denemeler:
        url = f"https://generativelanguage.googleapis.com/{versiyon}/models/{model}:generateContent?key={api_key}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens}
        }
        try:
            r = requests.post(url, json=body, timeout=30)
            print(f"  {versiyon}/{model} -> HTTP {r.status_code}")
            d = r.json()
            if "candidates" in d and d["candidates"]:
                print(f"  Basarili: {versiyon}/{model}")
                return d["candidates"][0]["content"]["parts"][0]["text"]
            if "error" in d:
                print(f"  Hata: {d['error'].get('message', '')[:100]}")
        except Exception as e:
            print(f"  {versiyon}/{model} exception: {e}")
            continue

    return "Gemini yanit vermedi."

def ai_tavsiye(analizler, doviz):
    veri = "\n".join([
        f"{a['ticker']}: {a['fiyat']} TL ({'+' if a['degisim']>0 else ''}{a['degisim']}%) | "
        f"RSI:{a['rsi']} | MACD hist:{a['macd_hist']} | Hacim:x{a['hacim_orani']} | "
        f"MA20:{a['ma20']} MA50:{a['ma50']} | BB:%{a['bb_poz']} | "
        f"Destek:{a['destek']} Direnc:{a['direnc']} | "
        f"Sinyaller:[{', '.join(a['sinyaller']) or 'yok'}]"
        for a in analizler
    ])

    kur_satir = " | ".join([f"{k}: {v}" for k, v in doviz.items()])

    prompt = f"""Sen deneyimli bir Turk borsa analistisin. Gercek teknik analiz verilerini yorumluyorsun.

PIYASA VERILERI:
{kur_satir}

HISSE VERILERI:
{veri}

Her hisse icin su formatta yaz:

TICKER - AL / SAT / TUT
Gerekcе: 2 cumle teknik gerekcе, RSI/MACD/MA verilerini kullan.
Hedef: yaklasik X TL | Stop: yaklasik Y TL
Risk: 1 cumle risk faktoru.

Son olarak 2-3 cumle BIST genel degerlendirmesi yaz.

Not: Teknik verilere sadik kal. Spekulatif olma.
Bu analiz yatirim tavsiyesi degildir."""

    return gemini_istek(prompt, 2000)

def ai_trend_yorum(trendler):
    if not trendler:
        return "Belirgin trend hisse bulunamadi."
    liste = "\n".join([
        f"{h['ticker']}: {'+' if h['gun_deg']>0 else ''}{h['gun_deg']:.1f}% gunluk, "
        f"haftalik {'+' if h['hafta_deg']>0 else ''}{h['hafta_deg']:.1f}%, hacim x{h['hacim_r']:.1f}"
        for h in trendler
    ])
    prompt = f"""BIST'te bugün one cikan hisseler (hacim + momentum bazli):

{liste}

3-4 cumle ile yorumla: Hangileri one cikiyor? Sektorel tema var mi? Kisa vadeli firsat var mi?"""
    return gemini_istek(prompt, 500)

# ─── TELEGRAM ─────────────────────────────────────────────────────────────────

def telegram_gonder(mesaj):
    token = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("Telegram bilgileri eksik, mesaj yazdiriliyor:")
        print(mesaj)
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    parcalar = [mesaj[i:i+4000] for i in range(0, len(mesaj), 4000)]

    for parca in parcalar:
        try:
            r = requests.post(url, json={
                "chat_id": chat_id,
                "text": parca,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }, timeout=15)
            if r.status_code == 200:
                print(f"  Telegram: gonderildi ({len(parca)} karakter)")
            else:
                print(f"  Telegram hata: {r.text}")
        except Exception as e:
            print(f"  Telegram baglanti hatasi: {e}")
        time.sleep(0.5)

# ─── RAPOR ────────────────────────────────────────────────────────────────────

def rapor_olustur(analizler, doviz, ai_yorum, trendler, trend_yorum):
    tarih = datetime.now().strftime("%d.%m.%Y %A")
    saat = datetime.now().strftime("%H:%M")

    kur_satir = ""
    if doviz.get("USD/TRY"):
        kur_satir += f"USD: {doviz['USD/TRY']} TL"
    if doviz.get("EUR/TRY"):
        kur_satir += f"  |  EUR: {doviz['EUR/TRY']} TL"
    if doviz.get("ALTIN_GRAM_TL"):
        kur_satir += f"  |  Altin: {int(doviz['ALTIN_GRAM_TL'])} TL/gr"

    al_list = [a for a in analizler if a["karar"] == "AL"]
    sat_list = [a for a in analizler if a["karar"] == "SAT"]
    tut_list = [a for a in analizler if a["karar"] == "TUT"]

    ozet = ""
    if al_list:
        ozet += "🟢 AL: " + " · ".join([a["ticker"] for a in al_list]) + "\n"
    if sat_list:
        ozet += "🔴 SAT: " + " · ".join([a["ticker"] for a in sat_list]) + "\n"
    if tut_list:
        ozet += "🟡 TUT: " + " · ".join([a["ticker"] for a in tut_list]) + "\n"

    trend_satir = ""
    for h in trendler[:6]:
        yon = "📈" if h["gun_deg"] >= 0 else "📉"
        trend_satir += f"{yon} {h['ticker']}: {h['fiyat']} TL | {'+' if h['gun_deg']>0 else ''}{h['gun_deg']:.1f}% | Hacim x{h['hacim_r']:.1f}\n"

    rapor = f"""📊 <b>GÜNLÜK BORSA RAPORU</b>
📅 {tarih} — ⏰ {saat}
━━━━━━━━━━━━━━━━━━━━━━━━

💱 <b>PİYASA</b>
{kur_satir}

━━━━━━━━━━━━━━━━━━━━━━━━
📋 <b>ÖZET SİNYALLER</b>
{ozet.strip()}

━━━━━━━━━━━━━━━━━━━━━━━━
🔥 <b>TREND HİSSELER</b>
{trend_satir.strip()}

{trend_yorum}

━━━━━━━━━━━━━━━━━━━━━━━━
🤖 <b>AI ANALİZ VE TAVSİYELER</b>
{ai_yorum}

━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ <i>Bu rapor yatırım tavsiyesi değildir.</i>
🤖 Gemini 1.5 Flash + Yahoo Finance
⏱️ {saat} | BORSA.AI"""

    return rapor

# ─── ANA FONKSİYON ────────────────────────────────────────────────────────────

def main():
    print(f"\n BIST AI Ajan baslatildi — {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print("=" * 55)

    print("\n Doviz verileri aliniyor...")
    doviz = doviz_cek()
    print(f"  USD/TRY: {doviz.get('USD/TRY', '?')} | EUR/TRY: {doviz.get('EUR/TRY', '?')}")

    print(f"\n {len(TAKIP_LISTESI)} hisse analiz ediliyor...")
    analizler = []
    for ticker in TAKIP_LISTESI:
        print(f"  -> {ticker}", end=" ", flush=True)
        a = hisse_analiz(ticker)
        if a:
            analizler.append(a)
            yon = "+" if a["degisim"] >= 0 else ""
            print(f"ok  {a['fiyat']} TL  ({yon}{a['degisim']}%)  RSI:{a['rsi']}  {a['karar']}")
        else:
            print("veri yok")
        time.sleep(0.3)

    print(f"\n  {len(analizler)}/{len(TAKIP_LISTESI)} hisse hazir")

    print("\n Trend hisseler taranıyor...")
    trendler = trend_tara()

    print("\n Gemini AI analiz yapiyor...")
    ai_yorum = ai_tavsiye(analizler, doviz)
    print("  Hisse yorumu hazir")

    trend_yorum = ai_trend_yorum(trendler)
    print("  Trend yorumu hazir")

    print("\n Rapor olusturuluyor...")
    rapor = rapor_olustur(analizler, doviz, ai_yorum, trendler, trend_yorum)

    print("\n Telegram'a gonderiliyor...")
    telegram_gonder(rapor)

    print("\n Tamamlandi!\n")

if __name__ == "__main__":
    main()
