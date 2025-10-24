# RiskX API Endpoints

## Complete Endpoint Reference

This document provides a comprehensive list of all available API endpoints.

### `/`

**GET** - Root endpoint

Get platform information and status

*Tags: health*

**Responses:**
- `200`: Platform information

---

### `/api/v1/health`

**GET** - Health check

Comprehensive system health check

*Tags: health*

**Responses:**
- `200`: System health status

---

### `/api/v1/risk/overview`

**GET** - Get risk overview

Comprehensive risk assessment from all ML models

*Tags: risk*

**Responses:**
- `200`: Risk overview data

---

### `/api/v1/risk/predictions/recession`

**GET** - Recession probability prediction

Get recession probability from ML model

*Tags: risk*

**Responses:**
- `200`: Recession prediction

---

### `/api/v1/risk/predictions/supply-chain`

**GET** - Supply chain risk prediction

Get supply chain risk assessment from ML model

*Tags: risk*

**Responses:**
- `200`: Supply chain risk prediction

---

### `/api/v1/risk/predictions/market-volatility`

**GET** - Market volatility prediction

Get market volatility prediction from ML model

*Tags: risk*

**Responses:**
- `200`: Market volatility prediction

---

### `/api/v1/risk/predictions/geopolitical`

**GET** - Geopolitical risk prediction

Get geopolitical risk prediction from ML model

*Tags: risk*

**Responses:**
- `200`: Geopolitical risk prediction

---

### `/api/v1/economic/indicators`

**GET** - Get economic indicators

Retrieve current economic indicators from multiple sources

*Tags: economic*

**Responses:**
- `200`: Economic indicators data

---

### `/api/v1/network/simulation`

**POST** - Run risk simulation

Execute shock propagation simulation on risk network

*Tags: network*

**Request Body Required**

**Responses:**
- `200`: Simulation results

---

### `/api/v1/cache/metrics`

**GET** - Cache performance metrics

Get cache system performance metrics

*Tags: cache*

**Responses:**
- `200`: Cache metrics

---

