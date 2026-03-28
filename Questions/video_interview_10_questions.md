# DeforestNet — Final 10 Interview Questions (Competitor-Aligned)

> Each question is designed so the officer's answer **directly exposes a gap in existing tools** and validates the **exact feature DeforestNet will build.**

---

## Q1. Have you ever received any automated deforestation alert from the government — like Anavaran or FSI?

**Competitor Gap:** Anavaran sends alerts to state HQ only. Field officers never receive them.

**Suggested Answer:**
> "No. I've heard such systems exist at the central level, but **at my level — range officer, beat guard — we have never received any automated alert**. Whatever information comes, it goes to the Conservator or DFO office. By the time it trickles down to us through official channels, it's already **weeks old and useless**. We still rely entirely on our own patrols and local informants."

**What We Commit to Build:**
- ✅ **Direct-to-officer mobile alert system** — bypasses HQ, reaches field staff
- ✅ Push notification + SMS fallback for low-connectivity areas
- ✅ Alert in **local language** (Hindi, Kannada, etc.)

---

## Q2. How quickly do you need a deforestation alert to be useful? Is every 15 days enough?

**Competitor Gap:** Anavaran was fortnightly (15 days). GFW/GLAD is weekly. Both too slow.

**Suggested Answer:**
> "15 days is far **too late**. Timber smugglers can clear 5–10 hectares in **2–3 nights**. By the 15th day, the wood is sold, the stumps are burned, and sometimes **crops are already planted** on the cleared land. Even weekly is often too slow. For it to be truly useful, I'd need alerts within **3–5 days** of the actual event. The faster, the more trees we can save."

**What We Commit to Build:**
- ✅ **5-day alert cycle** using Sentinel-1 (6-day revisit) + Sentinel-2 (5-day revisit)
- ✅ Combined SAR + optical processing to maximize detection frequency
- ✅ Priority queue: high-probability alerts pushed within **24 hours** of satellite pass

---

## Q3. During monsoon (June–September), can any existing system help you monitor forests?

**Competitor Gap:** GFW/GLAD is optical-only — completely blind during monsoon. Anavaran claimed SAR but was paused.

**Suggested Answer:**
> "Nothing works during monsoon. I've checked websites like Global Forest Watch — during cloudy months, the images are just **white patches, completely useless**. Our own aerial surveys stop. Patrolling drops 70%. And the smugglers know this — **monsoon is their peak season**. We have a saying: *'Baarish mein jungle sahab ka nahi, mafia ka hota hai'* — In the rains, the forest belongs to the mafia, not the officer. For 4 months every year, we are **completely blind**."

**What We Commit to Build:**
- ✅ **Sentinel-1 SAR processing pipeline** — already in our system (VV, VH bands)
- ✅ SAR-based detection runs **year-round**, independent of cloud cover
- ✅ Monsoon-specific model: SAR-only inference mode when optical data is unavailable
- ✅ This is our **#1 technical differentiator** over GFW, GLAD, and any optical-only system

---

## Q4. Can you detect small-scale tree felling — say 50 trees or 0.5 hectares — with any existing tool?

**Competitor Gap:** DETER misses anything <25ha. Anavaran had no published minimum. GFW at 30m resolution misses sub-hectare patches.

**Suggested Answer:**
> "Absolutely not. The government survey reports talk about forest cover in **thousands of hectares** — they can't detect 50 trees missing. And the smugglers have learned this. They don't clear large areas anymore. They do **selective felling — 20 trees here, 30 trees there**, spread across kilometers. Each patch is less than 1 hectare. No satellite report has ever caught these. But if you add them up over a year in my range alone, it's probably **200–300 hectares** of degradation that's completely invisible."

**What We Commit to Build:**
- ✅ **10m resolution pixel-level segmentation** — each pixel = 10m × 10m = 100 sq m
- ✅ 256×256 patch analysis detects changes as small as **0.5 hectares**
- ✅ 11-band feature stack (NDVI, EVI, SAVI, VV/VH, RVI) catches subtle vegetation loss
- ✅ Class-balanced training (73.1% / 26.9%) ensures rare small-patch events aren't ignored

---

## Q5. When you find deforestation, can you determine WHY it happened — logging vs. mining vs. farming vs. fire?

**Competitor Gap:** ALL existing systems (GFW, Anavaran, DETER) only say "forest loss." None classify the CAUSE.

**Suggested Answer:**
> "On the ground, I can tell — if there are stumps it's logging, if there's a pit it's mining, if there are crops it's encroachment. But from any satellite report or government data? They just say **'forest cover decreased.'** The cause is never mentioned. And this matters legally — **penalties are different** for mining (₹25 lakh+) vs. farming encroachment (₹5,000). Different departments handle each type. Without knowing the cause, we can't even file the right case."

**What We Commit to Build (Phase 2):**
- ✅ **Multi-class classification** — expand from binary (deforestation/not) to 5 classes:
  - Illegal logging
  - Agricultural encroachment
  - Mining activity
  - Forest fire damage
  - Road/infrastructure clearing
- ✅ This feature **doesn't exist in ANY competing system** — first-of-its-kind

---

## Q6. How do you currently prepare evidence for legal cases against forest crimes?

**Competitor Gap:** No existing system generates court-ready evidence. Officers use handwritten reports and personal phone photos.

**Suggested Answer:**
> "Completely manual. I go to the site, **walk the perimeter**, estimate the area by guessing — 'maybe 3 hectares, maybe 5.' I take photos on my personal phone — no timestamp proof, no GPS embedded. I write a **panchnama** (ground report) by hand. The whole process takes **2–3 days per case**. And in court, defense lawyers regularly challenge our evidence: *'Where's the satellite proof? Where's the precise measurement? This is just your opinion.'* We've had **cases dismissed** because our evidence wasn't considered scientific enough."

**What We Commit to Build:**
- ✅ **Auto-generated evidence package** per alert:
  - GPS coordinates (lat/long) of affected area
  - Before-and-after satellite imagery with timestamp
  - Precise area measurement in hectares (pixel count × 100 sq m)
  - AI confidence score (e.g., "92% probability of deforestation")
  - GradCAM heatmap showing WHAT the model detected
- ✅ **Exportable as PDF** — ready for FIR, panchnama, and court submission
- ✅ This replaces 2–3 days of manual work with a **5-minute download**

---

## Q7. If an AI system flagged deforestation, would you trust it? What proof would you need?

**Competitor Gap:** No system provides explainability. Officers don't trust "black box" AI. Anavaran had no transparency.

**Suggested Answer:**
> "I wouldn't trust just a red dot on a map. If I take action based on a wrong alert — raid a village, mobilize police — and it turns out to be nothing, **I face serious backlash** from locals and even inquiry from superiors. I need to SEE why the system thinks it's deforestation. Show me the **before image and after image** side by side. Show me **how sure the system is** — 90% sure or 50% sure. If it's 90% with clear image proof, **I'll act immediately**. If it's 60% with unclear imagery, **I'll verify first**. Give me the choice."

**What We Commit to Build:**
- ✅ **Explainable AI dashboard:**
  - Before/after satellite image comparison
  - AI confidence percentage per alert
  - **GradCAM/attention heatmap overlay** — highlights exactly which pixels triggered the alert
  - Color-coded alert levels: 🔴 High confidence (>85%) → Act now, 🟡 Medium (60–85%) → Verify, 🟢 Low (<60%) → Monitor
- ✅ Officer can mark each alert as **"Verified" or "False Alarm"** → feedback loop improves the model

---

## Q8. Are there zones in your range where deforestation keeps recurring? Can you predict where it'll happen next?

**Competitor Gap:** No system offers predictive mapping. All existing tools are 100% reactive.

**Suggested Answer:**
> "Yes — I can tell you exactly: **near the state highway**, **along the river bank**, and **where the village boundary meets the reserved forest**. These 3–4 spots see activity every year. I know the patterns from 10 years of experience. But there's no data system that tracks this. The moment I reduce patrols at one hotspot to cover another, **activity shifts right back**. I wish I had a system that could tell me: *'This month, focus on Zone 3 — risk is highest there'* — so I can use my limited team smartly."

**What We Commit to Build (Phase 2):**
- ✅ **Predictive deforestation risk map:**
  - Analyze 5+ years of historical satellite data
  - Factor in: road proximity, river distance, village boundary, terrain slope, season, past events
  - Generate monthly **risk heatmap** per range
  - AI-recommended patrol routes for optimal coverage
- ✅ Shifts from **reactive** (respond after damage) to **preventive** (deploy before damage)
- ✅ **No competitor does this** — first-of-its-kind for Indian forests

---

## Q9. Does any existing tool work on your phone? Can you use it in the field with poor connectivity?

**Competitor Gap:** GFW needs stable internet + desktop browser. Anavaran had no field app. Academic tools have no interface at all.

**Suggested Answer:**
> "I have a basic Android smartphone — maybe ₹10,000–15,000. Internet works in town but in the forest, I barely get **2G or sometimes no signal at all**. Most of the websites I've been shown — Global Forest Watch, ISRO portals — they **don't even load properly** on my phone. And in the field where I actually need the information, there's no connectivity to check anything. If a tool doesn't work on **my phone, in my language, with poor internet**, it's useless to me."

**What We Commit to Build:**
- ✅ **Lightweight mobile app** (Android) — works on phones with 2GB RAM
- ✅ **Three-tier alert delivery:**
  - Tier 1: App push notification (3G/4G areas)
  - Tier 2: Compressed SMS with GPS coordinates (2G areas)
  - Tier 3: WhatsApp message with map image (most widely used)
- ✅ **Offline mode** — last 50 alerts cached locally, viewable without internet
- ✅ **Multi-language support** — Hindi, Kannada, Tamil, Telugu, Malayalam, Bengali
- ✅ All processing happens **on cloud** — phone only receives results

---

## Q10. If we build everything you've described — alerts, evidence, predictions — what would make you actually USE it every day?

**Competitor Gap:** Adoption failure. Anavaran was built but nobody used it. Technology alone doesn't guarantee adoption.

**Suggested Answer:**
> "Three things. **First, simplicity** — if it needs training or a manual, my guards won't touch it. It should be as easy as reading a WhatsApp message. **Second, accuracy** — if it sends 10 alerts and 5 are wrong, I'll stop trusting it within a month. Better to send fewer alerts that are accurate. **Third, it should make my REPORTING easier** — if the same system that sends alerts can also generate my monthly report automatically, you save me 3–4 days per month. **That** would make me use it every day, because it's not extra work — it replaces my hardest work."

**What We Commit to Build:**
- ✅ **WhatsApp-simple interface** — no training needed
- ✅ **High-precision mode** — configurable threshold (e.g., only alert when confidence >85%)
- ✅ **Auto-generated monthly reports:**
  - Total area monitored
  - Number of deforestation events detected
  - Total hectares affected
  - Before/after imagery per event
  - Trend comparison with previous months
  - Export as PDF for submission to DFO/Conservator
- ✅ **Officer feedback loop** — "Verified" / "False Alarm" buttons train the model over time

---

## Summary: Interview → Gap → Deliverable

| Q# | Competitor Gap Exposed | Exact DeforestNet Deliverable | Phase |
|----|----------------------|------------------------------|:-----:|
| 1 | Anavaran never reaches field officers | Direct mobile alert to officer | 1 |
| 2 | 15-day / weekly cycles too slow | 5-day alert cycle | 1 |
| 3 | Optical systems blind during monsoon | SAR-based year-round monitoring | 1 |
| 4 | Can't detect <25ha patches | 10m pixel-level, 0.5ha detection | 1 |
| 5 | No system classifies cause of deforestation | Multi-class: logging/mining/fire/farming | 2 |
| 6 | No court-ready evidence system | Auto-generated evidence PDF | 1 |
| 7 | No explainability in any system | GradCAM + confidence scores + feedback | 1 |
| 8 | All systems are reactive | Predictive risk heatmaps | 2 |
| 9 | No tool works on basic phones / offline | Lightweight app + SMS + WhatsApp + offline | 1 |
| 10 | Tools built but nobody uses them | Simple UX + auto-reports + feedback loop | 1 |
