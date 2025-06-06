# System Insights

## Overview
A real-time monitoring solution that collects and visualizes system metrics including CPU, memory, disk, network, and GPU usage.

![Dashboard Preview](assets/dashboard-preview.png)

## ✨ Features
- **Real-time Monitoring**
  - CPU usage and load averages
  - Memory utilization (RAM and swap)
  - Disk I/O operations and space usage
  - Network traffic and active connections
  - GPU load and temperature monitoring

- **Multi-system Support**
  - Monitor multiple machines from single dashboard
  - System comparison views

## 🛠️ Installation

### Setup

1. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Configure .env
   Insert MongoDB credentials

## 🚀 Usage

1. Data Collection

Run on monitored systems:
```bash
python main.py
```

2. Dashboard

Start the visualization interface:
```bash
streamlit run dashboard.py
```
    

