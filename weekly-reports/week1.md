# DeforestNet — Weekly Report

## Week 1: Problem Identification & Research

**Project:** Deforestation Detection & Alert System (DeforestNet)  
**Week:** Week 1 (18 March – 25 March 2026)  
**Team:** [Add your team member names here]

---

## Tasks Completed

### 1. Problem Statement Identification

- Studied India's deforestation crisis — **1,50,000 hectares lost per year**
- Identified core gap: no AI monitoring tool reaches field-level forest officers
- Analyzed existing solutions and their limitations:
  - **Anavaran (FSI)** — paused since Oct 2025; alerts only reached state HQ, not field officers
  - **Global Forest Watch** — global focus, not India-specific, no mobile alerts
  - **GLAD Alerts** — optical only, fails during 4-month monsoon season
  - **DETER (Brazil)** — Brazil-only; misses patches smaller than 25 hectares
- **Finalized Problem Statement:**  
  > *Forest officers responsible for 100–300 sq km of forest have no real-time, all-weather monitoring tool — deforestation goes undetected for weeks, and evidence collected manually is too weak for legal action.*

### 2. Team Structure & Setup

- Defined team roles and responsibilities
- Set up project communication channels
- Created GitHub repository: [DeforestNet](https://github.com/prajwal-tech07/DeforestNet)
- Established project folder structure

### 3. Research & Literature Survey

- **Satellite Data:** Studied Sentinel-1 (SAR radar) and Sentinel-2 (optical) imagery from ESA
- **Deep Learning:** Researched UNet architecture for binary semantic segmentation
- **Spectral Indices:** NDVI, EVI, SAVI, VV/VH ratio, RVI — 11 feature bands total
- **Dataset:** Identified Brazilian Amazon dataset (Sentinel-1 & 2 with ground truth masks)
- **Region:** EPSG:32722 — UTM zone 22S, 16 patches of 2816×2816 pixels each

### 4. Competitive Analysis

| Competitor | Weakness | Our Edge |
|---|---|---|
| Anavaran (FSI) | Paused; top-down only | Bottom-up — alerts reach field officers |
| Global Forest Watch | Not India-specific; no mobile alerts | India-focused; phone-based alerts |
| GLAD Alerts | Optical only — fails in monsoon | SAR radar — works through clouds |
| DETER (Brazil) | Brazil-only; 25ha minimum | Detects changes ≥ 0.5 hectares |

### 5. Product Definition

- **Ideal Customer Profile (ICP):** Indian Forest Range Officers & Beat Guards
- **Product USP:**
  1. All-weather monitoring (SAR + Optical fusion)
  2. Sub-hectare detection at 10m resolution
  3. Built for field officers, not headquarters
  4. AI-generated court-ready evidence

---

## Summary

| Activity | Status |
|---|---|
| Problem statement finalized | ✅ Done |
| Team roles assigned | ✅ Done |
| GitHub repo created | ✅ Done |
| Competitor analysis completed | ✅ Done |
| Literature survey (UNet, SAR, Sentinel) | ✅ Done |
| Dataset identified | ✅ Done |
| ICP defined | ✅ Done |
| Product USP defined | ✅ Done |

---

## Plan for Week 2

- [ ] Download and organize Sentinel-1 & Sentinel-2 dataset
- [ ] Set up Python virtual environment and install dependencies
- [ ] Begin preprocessing pipeline — GeoTIFF reading, noise removal, normalization
- [ ] Implement feature extraction (NDVI, EVI, SAVI, SAR indices)
- [ ] Prepare 256×256 patch extraction with class balancing

---

## Challenges

- Large satellite dataset requires stable internet for download
- GPU access needed for model training in upcoming weeks
