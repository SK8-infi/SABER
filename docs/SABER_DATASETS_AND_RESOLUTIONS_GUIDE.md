# SABER: Spatial Resolution & Dataset Specifications Guide
### Sensor-Agnostic Bridged Embedding Retrieval (ISRO BAH 2026 · Problem Statement 11)

---

## 📌 Overview

This document presents a scientific reference of the dataset spatial resolutions, Ground Sampling Distances (GSD), patch pixel dimensions, spectral band wavelengths, and ground physical area coverage for the two core multi-sensor benchmark datasets supported by SABER:

1. **BigEarthNet BEN-14K** (Sentinel-1 Synthetic Aperture Radar ◄► Sentinel-2 Multispectral)
2. **DSRSID** (Gaofen-1 Panchromatic ◄► Gaofen-1 Multispectral)

---

## 🛰️ 1. BigEarthNet BEN-14K Dataset Specifications

### A. General Dataset & Satellite Characteristics
* **Satellites**: European Space Agency (ESA) **Sentinel-1** (Constellation 1A/1B) and **Sentinel-2** (Constellation 2A/2B).
* **Patch Pixel Matrix Size**: $120 \times 120$ pixels per patch (interpolated/resized to $224 \times 224$ for Vision Transformer input).
* **Ground Coverage Area**: $1.2\,\text{km} \times 1.2\,\text{km}$ ($1200\,\text{m} \times 1200\,\text{m}$ physical area on Earth's surface per patch).
* **Target Classes**: 19 multi-hot land-cover nomenclature classes derived from CORINE Land Cover (CLC).

### B. Sensor-Specific Resolution Breakdown

| Modality / Sensor | Band Name | Central Wavelength ($\lambda_c$) | Raw Patch Size | Ground Sampling Distance (GSD) | Spectral Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Sentinel-1 SAR** | Band 1 (VV) | $5.405\,\mu\text{m}$ (C-band) | $120 \times 120$ | **$10\,\text{m / pixel}$** | Vertical transmit / Vertical receive polarization |
| **Sentinel-1 SAR** | Band 2 (VH) | $5.405\,\mu\text{m}$ (C-band) | $120 \times 120$ | **$10\,\text{m / pixel}$** | Vertical transmit / Horizontal receive polarization |
| **Sentinel-2 MS** | Band 2 (Blue) | $0.490\,\mu\text{m}$ | $120 \times 120$ | **$10\,\text{m / pixel}$** | Visible Blue spectrum |
| **Sentinel-2 MS** | Band 3 (Green) | $0.560\,\mu\text{m}$ | $120 \times 120$ | **$10\,\text{m / pixel}$** | Visible Green spectrum |
| **Sentinel-2 MS** | Band 4 (Red) | $0.665\,\mu\text{m}$ | $120 \times 120$ | **$10\,\text{m / pixel}$** | Visible Red spectrum |
| **Sentinel-2 MS** | Band 5 (Red Edge 1) | $0.705\,\mu\text{m}$ | $120 \times 120$ | **$20\,\text{m / pixel}$** *(resampled)* | Vegetation Red Edge |
| **Sentinel-2 MS** | Band 6 (Red Edge 2) | $0.740\,\mu\text{m}$ | $120 \times 120$ | **$20\,\text{m / pixel}$** *(resampled)* | Vegetation Red Edge |
| **Sentinel-2 MS** | Band 7 (Red Edge 3) | $0.783\,\mu\text{m}$ | $120 \times 120$ | **$20\,\text{m / pixel}$** *(resampled)* | Vegetation Red Edge |
| **Sentinel-2 MS** | Band 8 (NIR) | $0.842\,\mu\text{m}$ | $120 \times 120$ | **$10\,\text{m / pixel}$** | Near-Infrared |
| **Sentinel-2 MS** | Band 8A (Narrow NIR)| $0.865\,\mu\text{m}$ | $120 \times 120$ | **$20\,\text{m / pixel}$** *(resampled)* | Narrow Near-Infrared |
| **Sentinel-2 MS** | Band 9 (Water Vapor) | $0.945\,\mu\text{m}$ | $120 \times 120$ | **$60\,\text{m / pixel}$** *(resampled)* | Water Vapour absorption |
| **Sentinel-2 MS** | Band 11 (SWIR 1) | $1.610\,\mu\text{m}$ | $120 \times 120$ | **$20\,\text{m / pixel}$** *(resampled)* | Short-Wave Infrared 1 |
| **Sentinel-2 MS** | Band 12 (SWIR 2) | $2.190\,\mu\text{m}$ | $120 \times 120$ | **$20\,\text{m / pixel}$** *(resampled)* | Short-Wave Infrared 2 |

---

## 🛰️ 2. DSRSID Dataset Specifications

### A. General Dataset & Satellite Characteristics
* **Satellite**: China National Space Administration (CNSA) **Gaofen-1 (GF-1)** Optical Satellite.
* **Ground Coverage Area**: $640\,\text{m} \times 640\,\text{m}$ ($0.64\,\text{km} \times 0.64\,\text{km}$ physical area on Earth per scene patch).
* **Target Classes**: 8 single-label land-use categories (Aquafarm, Cloud, Forest, High Building, Low Building, Farm Land, River, Water).

### B. Sensor-Specific Resolution Breakdown

| Modality / Sensor | Channels | Raw Patch Size | Spatial Ground Sampling Distance (GSD) | Central Wavelength ($\lambda_c$) | Spectral Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gaofen-1 Panchromatic (PAN)** | 1 | **$256 \times 256$** | **$2.5\,\text{meters / pixel}$** | $0.675\,\mu\text{m}$ | High-Resolution Single-Band Panchromatic |
| **Gaofen-1 Multispectral (MS)** | 4 | **$64 \times 64$** | **$10.0\,\text{meters / pixel}$** | $0.485\,\mu\text{m} - 0.830\,\mu\text{m}$ | Blue, Green, Red, Near-Infrared |

> **Note on Spatial Aspect Ratio**: Notice that the ratio of patch pixel matrix sizes ($256 / 64 = 4$) matches the $4\times$ ground sampling resolution ratio between Panchromatic ($2.5\,\text{m}$) and Multispectral ($10.0\,\text{m}$) sensors:
> $$\text{Physical Field of View} = 256 \text{ pixels} \times 2.5\,\text{m/pixel} = 64 \text{ pixels} \times 10.0\,\text{m/pixel} = 640\,\text{meters}$$

---

## 📊 Summary Comparison Matrix

| Dataset Metric | BEN-14K (Sentinel-1 SAR) | BEN-14K (Sentinel-2 MS) | DSRSID (Gaofen-1 PAN) | DSRSID (Gaofen-1 MS) |
| :--- | :---: | :---: | :---: | :---: |
| **Active Channels** | 2 | 12 | 1 | 4 |
| **Raw Patch Dimensions** | $120 \times 120$ | $120 \times 120$ | $256 \times 256$ | $64 \times 64$ |
| **SABER Input Dimensions** | $224 \times 224 \times 2$ | $224 \times 224 \times 12$ | $224 \times 224 \times 1$ | $224 \times 224 \times 4$ |
| **Ground Sampling Distance (GSD)** | **$10\,\text{m / pixel}$** | **$10\,\text{m} / 20\,\text{m} / 60\,\text{m}$** | **$2.5\,\text{m / pixel}$** | **$10.0\,\text{m / pixel}$** |
| **Ground Coverage Area** | $1.2\,\text{km} \times 1.2\,\text{km}$ | $1.2\,\text{km} \times 1.2\,\text{km}$ | $640\,\text{m} \times 640\,\text{m}$ | $640\,\text{m} \times 640\,\text{m}$ |
| **Central Wavelengths ($\lambda_c$)** | $[5.405, 5.405]\,\mu\text{m}$ | $[0.443 - 2.190]\,\mu\text{m}$ | $[0.675]\,\mu\text{m}$ | $[0.485 - 0.830]\,\mu\text{m}$ |
