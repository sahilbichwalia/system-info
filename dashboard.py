import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from datetime import datetime, timedelta
import pytz
import numpy as np
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="System Monitoring Dashboard",
    page_icon="🖥️",
    layout="wide"
)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight {
        background-color: #e6f7ff;
        border-left: 4px solid #1890ff;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .status-healthy {
        color: #52c41a;
        font-weight: bold;
    }
    .status-warning {
        color: #faad14;
        font-weight: bold;
    }
    .status-critical {
        color: #f5222d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Connect to MongoDB
@st.cache_resource
def get_mongo_connection():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[DB_NAME]
        return db[COLLECTION_NAME]
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

collection = get_mongo_connection()

if collection is None:
    st.error("Could not establish database connection. Please check your MongoDB settings.")
    st.stop()

# Get unique systems
@st.cache_data(ttl=3600)
def get_systems():
    try:
        return collection.distinct("mac_address")
    except Exception as e:
        st.error(f"Error fetching systems: {str(e)}")
        return []

# Sidebar filters
st.sidebar.title("Dashboard Filters")
selected_system = st.sidebar.selectbox(
    "Select System",
    options=get_systems(),
    index=0 if get_systems() else None
)

# Date range filter
default_end = datetime.now(pytz.utc)
default_start = default_end - timedelta(days=1)
date_range = st.sidebar.date_input(
    "Date Range",
    value=(default_start.date(), default_end.date()),
    min_value=default_end.date() - timedelta(days=30),
    max_value=default_end.date()
)

# Time aggregation
agg_options = {
    "Raw Data": None,
    "1 Minute": "1min",
    "5 Minutes": "5min",
    "15 Minutes": "15min",
    "1 Hour": "1h"
}

selected_agg = st.sidebar.selectbox(
    "Time Aggregation",
    options=list(agg_options.keys()),
    index=2
)

# Main content
st.title("System Monitoring Dashboard")

if selected_system:
   # Load data based on selected system and date range
    @st.cache_data(ttl=60)
    def get_system_data(mac_address, start_date, end_date):
        query = {
            "mac_address": mac_address,
            "timestamp": {
                "$gte": datetime.combine(start_date, datetime.min.time()).replace(tzinfo=pytz.utc),
                "$lte": datetime.combine(end_date, datetime.max.time()).replace(tzinfo=pytz.utc)
            }
        }
        
        try:
            # Fetch data from MongoDB
            data = list(collection.find(query, {'_id': 0}).sort("timestamp", 1))
            
            # Ensure all documents have the same structure
            for doc in data:
                doc.setdefault('cpu', {})
                doc.setdefault('memory', {})
                doc.setdefault('disk', {'partitions': [{}]})
                doc.setdefault('network', {})
                doc.setdefault('gpu', [{}])
                doc.setdefault('processes', {})
                doc.setdefault('system_health', {})
                
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    df = get_system_data(selected_system, date_range[0], date_range[1])
    
    if not df.empty:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # System Overview
        st.subheader("System Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Get the latest system ID and OS information
            system_id = df.iloc[-1].get('system_id', 'N/A')
            os_info = f"{df.iloc[-1].get('os', 'N/A')} {df.iloc[-1].get('os_version', '')}"
            st.markdown(f"""
            <div class="metric-card">
                <h4>System ID</h4>
                <p>{system_id}</p>
                <h4>OS</h4>
                <p>{os_info}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Get the latest system boot time and uptime
            uptime_seconds = df.iloc[-1].get('system_health', {}).get('uptime_seconds', 0)
            uptime_str = str(timedelta(seconds=int(uptime_seconds))) if uptime_seconds else "N/A"
            boot_time = df.iloc[-1].get('system_health', {}).get('boot_time', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <h4>Uptime</h4>
                <p>{uptime_str}</p>
                <h4>Last Boot</h4>
                <p>{boot_time}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Get the latest CPU usage and frequency
            cpu_df = pd.json_normalize(df['cpu'])
            cpu_usage = cpu_df.iloc[-1].get('cpu_usage', 0)
            cpu_status = "status-healthy" if cpu_usage < 70 else "status-warning" if cpu_usage < 90 else "status-critical"
            cpu_freq = cpu_df.iloc[-1].get('cpu_frequency_mhz', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <h4>CPU Usage</h4>
                <p class="{cpu_status}">{cpu_usage:.1f}%</p>
                <h4>Frequency</h4>
                <p>{cpu_freq} MHz</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Get the latest memory usage and processes
            mem_df = pd.json_normalize(df['memory'])
            mem_usage = mem_df.iloc[-1].get('memory_usage_percent', 0)
            mem_status = "status-healthy" if mem_usage < 70 else "status-warning" if mem_usage < 90 else "status-critical"
            mem_used = mem_df.iloc[-1].get('memory_used_gb', 'N/A')
            mem_total = df.iloc[-1].get('ram_size_gb', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <h4>Memory Usage</h4>
                <p class="{mem_status}">{mem_usage:.1f}% ({mem_used:.1f}/{mem_total:.1f} GB)</p>
                <h4>Processes</h4>
                <p>{df.iloc[-1].get('processes', {}).get('total_processes', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

        # Main tabs
        tab1, tab2, tab3, tab4, tab5,tab6,tab7,tab8 = st.tabs(["CPU", "Memory", "Disk", "Network", "GPU","Top Processes","System Details","System Snapshot"])
        
        with tab1:
            st.subheader("CPU Metrics")
            
            # Prepare CPU data
            cpu_data = []
            for _, row in df.iterrows():
                cpu_dict = row.get('cpu', {})
                cpu_dict['timestamp'] = row['timestamp']
                cpu_data.append(cpu_dict)
            
            cpu_df = pd.DataFrame(cpu_data)
            
            if not cpu_df.empty:
                # Resample data if aggregation is selected
                if agg_options[selected_agg]:

                    cpu_df = cpu_df.set_index('timestamp')
                    numeric_cols = cpu_df.select_dtypes(include=[np.number]).columns
                    cpu_df = cpu_df[numeric_cols].resample(agg_options[selected_agg]).mean().reset_index()
                
                # CPU Usage Chart
                fig_cpu = px.line(
                    cpu_df,
                    x="timestamp",
                    y="cpu_usage",
                    title="CPU Usage (%)",
                    labels={"cpu_usage": "Usage (%)", "timestamp": "Time"}
                )
                st.plotly_chart(fig_cpu, use_container_width=True,key=f"cpu_usage_{selected_system}")
                
                # CPU Load Chart
                if 'load_1min' in cpu_df.columns:
                    fig_load = go.Figure()
                    fig_load.add_trace(go.Scatter(
                        x=cpu_df['timestamp'],
                        y=cpu_df['load_1min'],
                        name="1 min",
                        line=dict(color='#1f77b4')
                    ))
                    if 'load_5min' in cpu_df.columns:
                        fig_load.add_trace(go.Scatter(
                            x=cpu_df['timestamp'],
                            y=cpu_df['load_5min'],
                            name="5 min",
                            line=dict(color='#ff7f0e')
                        ))
                    if 'load_15min' in cpu_df.columns:
                        fig_load.add_trace(go.Scatter(
                            x=cpu_df['timestamp'],
                            y=cpu_df['load_15min'],
                            name="15 min",
                            line=dict(color='#2ca02c')
                        ))
                    fig_load.update_layout(
                        title="CPU Load Average",
                        xaxis_title="Time",
                        yaxis_title="Load"
                    )
                    st.plotly_chart(fig_load, use_container_width=True,key=f"cpu_load_{selected_system}")
        
        with tab2:
            st.subheader("Memory Metrics")
            
            # Prepare memory data
            mem_data = []
            for _, row in df.iterrows():
                mem_dict = row.get('memory', {})
                mem_dict['timestamp'] = row['timestamp']
                mem_data.append(mem_dict)
            
            mem_df = pd.DataFrame(mem_data)
            
            if not mem_df.empty:
                if agg_options[selected_agg]:
                    mem_df = mem_df.set_index('timestamp')
                    numeric_cols = mem_df.select_dtypes(include=[np.number]).columns
                    mem_df = mem_df[numeric_cols].resample(agg_options[selected_agg]).mean().reset_index()
                
                # Memory Usage Chart
                fig_mem = px.line(
                    mem_df,
                    x="timestamp",
                    y="memory_usage_percent",
                    title="Memory Usage (%)",
                    labels={"memory_usage_percent": "Usage (%)", "timestamp": "Time"}
                )
                st.plotly_chart(fig_mem, use_container_width=True, key=f"mem_usage_{selected_system}")
                
                # Memory Breakdown Chart
                if 'memory_used_gb' in mem_df.columns and 'memory_available_gb' in mem_df.columns:
                    fig_mem_breakdown = go.Figure()
                    fig_mem_breakdown.add_trace(go.Scatter(
                        x=mem_df['timestamp'],
                        y=mem_df['memory_used_gb'],
                        fill='tozeroy',
                        name="Used",
                        line=dict(color='#d62728')
                    ))
                    fig_mem_breakdown.add_trace(go.Scatter(
                        x=mem_df['timestamp'],
                        y=mem_df['memory_available_gb'],
                        fill='tonexty',
                        name="Available",
                        line=dict(color='#1f77b4')
                    ))
                    fig_mem_breakdown.update_layout(
                        title="Memory Usage (GB)",
                        xaxis_title="Time",
                        yaxis_title="GB"
                    )
                    st.plotly_chart(fig_mem_breakdown, use_container_width=True, key=f"mem_breakdown_{selected_system}")
        
        with tab3:
            st.subheader("Disk Metrics")
            
            # Prepare disk data
            disk_data = []
            for _, row in df.iterrows():
                disk_dict = row.get('disk', {})
                disk_dict['timestamp'] = row['timestamp']
                disk_data.append(disk_dict)
            
            disk_df = pd.DataFrame(disk_data)
            
            if not disk_df.empty:
                if agg_options[selected_agg]:
                    disk_df = disk_df.set_index('timestamp')
                    numeric_cols = disk_df.select_dtypes(include=[np.number]).columns
                    disk_df = disk_df[numeric_cols].resample(agg_options[selected_agg]).mean().reset_index()
                
                # Disk Usage Chart
                partitions = []
                for _, row in df.iterrows():
                    for part in row.get('disk', {}).get('partitions', []):
                        part['timestamp'] = row['timestamp']
                        partitions.append(part)
                
                if partitions:
                    partitions_df = pd.DataFrame(partitions)
                    
                    fig_disk = px.line(
                        partitions_df,
                        x="timestamp",
                        y="usage_percent",
                        color="mountpoint",
                        title="Disk Usage by Partition (%)",
                        labels={"usage_percent": "Usage (%)", "timestamp": "Time"}
                    )
                    st.plotly_chart(fig_disk, use_container_width=True, key=f"disk_usage_{selected_system}")
                
                # Disk I/O Chart
                if 'total_read_mb' in disk_df.columns and 'total_write_mb' in disk_df.columns:
                    fig_disk_io = go.Figure()
                    fig_disk_io.add_trace(go.Scatter(
                        x=disk_df['timestamp'],
                        y=disk_df['total_read_mb'],
                        name="Read (MB)",
                        line=dict(color='#1f77b4')
                    ))
                    fig_disk_io.add_trace(go.Scatter(
                        x=disk_df['timestamp'],
                        y=disk_df['total_write_mb'],
                        name="Write (MB)",
                        line=dict(color='#ff7f0e')
                    ))
                    fig_disk_io.update_layout(
                        title="Disk I/O Activity",
                        xaxis_title="Time",
                        yaxis_title="MB"
                    )
                    st.plotly_chart(fig_disk_io, use_container_width=True, key=f"disk_io_{selected_system}")

            st.subheader("Disk Details")
            disks = df.iloc[-1].get('disks', [])
            for disk in disks:
                st.write(f"**{disk.get('device', 'Unknown')}**")
                st.write(f"Mount: {disk.get('mountpoint', 'N/A')}")
                st.write(f"Type: {disk.get('fstype', 'N/A')}")
                st.write(f"Size: {disk.get('total_gb', 'N/A')} GB")
        
        with tab4:
            st.subheader("Network Metrics")
            
            # Prepare network data
            net_data = []
            for _, row in df.iterrows():
                net_dict = row.get('network', {})
                net_dict['timestamp'] = row['timestamp']
                net_data.append(net_dict)
            
            net_df = pd.DataFrame(net_data)
            
            if not net_df.empty:
                if agg_options[selected_agg]:
                    net_df = net_df.set_index('timestamp')
                    numeric_cols = net_df.select_dtypes(include=[np.number]).columns
                    net_df = net_df[numeric_cols].resample(agg_options[selected_agg]).mean().reset_index()
                
                # Network Traffic Chart
                if 'bytes_sent_mb' in net_df.columns and 'bytes_recv_mb' in net_df.columns:
                    fig_net = go.Figure()
                    fig_net.add_trace(go.Scatter(
                        x=net_df['timestamp'],
                        y=net_df['bytes_sent_mb'],
                        name="Sent (MB)",
                        line=dict(color='#1f77b4')
                    ))
                    fig_net.add_trace(go.Scatter(
                        x=net_df['timestamp'],
                        y=net_df['bytes_recv_mb'],
                        name="Received (MB)",
                        line=dict(color='#ff7f0e')
                    ))
                    fig_net.update_layout(
                        title="Network Traffic",
                        xaxis_title="Time",
                        yaxis_title="MB"
                    )
                    st.plotly_chart(fig_net, use_container_width=True, key=f"net_traffic_{selected_system}")
                
                # TCP Connections Chart
                if 'active_connections' in net_df.columns:
                    fig_conn = px.line(
                        net_df,
                        x="timestamp",
                        y="active_connections",
                        title="Active TCP Connections",
                        labels={"active_connections": "Connections", "timestamp": "Time"}
                    )
                    st.plotly_chart(fig_conn, use_container_width=True, key=f"net_connections_{selected_system}")

            st.subheader("TCP Connection States")
            tcp_states = df.iloc[-1].get('network', {}).get('tcp_states', {})
            if tcp_states:
                fig_tcp = px.bar(
                    x=list(tcp_states.keys()),
                    y=list(tcp_states.values()),
                    labels={'x': 'State', 'y': 'Count'},
                    title="TCP Connection States"
                )
                st.plotly_chart(fig_tcp, use_container_width=True, key=f"tcp_states_{selected_system}")
    
            # Network Interfaces
            st.subheader("Network Interfaces")
            net_interfaces = df.iloc[-1].get('network_interfaces', [])
            for interface in net_interfaces:
                st.write(f"**{interface.get('name', 'Unknown')}:**")
                st.write(interface.get('addresses', []))
        
        with tab5:
            st.subheader("GPU Metrics")
            
            # Prepare GPU data
            gpu_data = []
            for _, row in df.iterrows():
                gpus = row.get('gpu', [{}])
                try:
                    for gpu in gpus:
                        gpu['timestamp'] = row['timestamp']
                        gpu_data.append(gpu)
                except Exception as e:
                    st.error(f"Error processing GPU data: {str(e)}")
            
            if gpu_data:
                gpu_df = pd.DataFrame(gpu_data)
                
                if not gpu_df.empty:
                    if agg_options[selected_agg]:
                        gpu_df = gpu_df.set_index('timestamp')
                        numeric_cols = gpu_df.select_dtypes(include=[np.number]).columns
                        gpu_df = gpu_df[numeric_cols].resample(agg_options[selected_agg]).mean().reset_index()
                    
                    # GPU Usage Chart
                    if 'load_percent' in gpu_df.columns:
                        fig_gpu = px.line(
                            gpu_df,
                            x="timestamp",
                            y="load_percent",
                            title="GPU Load (%)",
                            labels={"load_percent": "Load (%)", "timestamp": "Time"}
                        )
                        st.plotly_chart(fig_gpu, use_container_width=True, key=f"gpu_load_{selected_system}")
                    
                    # GPU Memory Chart
                    if 'memory_usage_percent' in gpu_df.columns:
                        fig_gpu_mem = px.line(
                            gpu_df,
                            x="timestamp",
                            y="memory_usage_percent",
                            title="GPU Memory Usage (%)",
                            labels={"memory_usage_percent": "Usage (%)", "timestamp": "Time"}
                        )
                        st.plotly_chart(fig_gpu_mem, use_container_width=True, key=f"gpu_mem_{selected_system}")
                    
                    # GPU Temperature Chart
                    if 'temperature' in gpu_df.columns:
                        fig_gpu_temp = px.line(
                            gpu_df,
                            x="timestamp",
                            y="temperature",
                            title="GPU Temperature (°C)",
                            labels={"temperature": "Temperature (°C)", "timestamp": "Time"}
                        )
                        st.plotly_chart(fig_gpu_temp, use_container_width=True, key=f"gpu_temp_{selected_system}")
            else:
                st.info("No GPU data available for this system")
        
        with tab6:
            # Processes section
            st.subheader("Top Processes")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top CPU Processes**")
                try:
                    top_cpu = df.iloc[-1].get('processes', {}).get('top_cpu_processes', [])
                    if top_cpu:
                        cpu_proc_df = pd.DataFrame(top_cpu)
                        st.dataframe(cpu_proc_df[['name', 'pid', 'cpu_percent', 'memory_percent', 'memory_rss_mb']], 
                                    height=300)
                    else:
                        st.info("No CPU process data available")
                except Exception as e:
                    st.error(f"Error displaying CPU processes: {str(e)}")
            
            with col2:
                st.markdown("**Top Memory Processes**")
                try:
                    top_mem = df.iloc[-1].get('processes', {}).get('top_memory_processes', [])
                    if top_mem:
                        mem_proc_df = pd.DataFrame(top_mem)
                        st.dataframe(mem_proc_df[['name', 'pid', 'memory_percent', 'memory_rss_mb', 'cpu_percent']], 
                                    height=300)
                    else:
                        st.info("No memory process data available")
                except Exception as e:
                    st.error(f"Error displaying memory processes: {str(e)}")
            
            # Raw data section
            with st.expander("View Raw Data"):
                # For display, we'll show a simplified version of the data
                display_df = df[['timestamp', 'system_id', 'cpu', 'memory', 'disk', 'network', 'gpu']].copy()
                st.dataframe(display_df, height=300)
                
                # For export, convert all data to strings to avoid serialization issues
                export_df = df.copy()
                for col in export_df.select_dtypes(include=['object']).columns:
                    export_df[col] = export_df[col].astype(str)
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"system_metrics_{selected_system}.csv",
                    mime="text/csv",
                    key=f"download_{selected_system}_{datetime.now().timestamp()}"  # Unique key
                        )
                
        with tab7:
            # System Details section
            st.subheader("System Details")
            st.write(f"**Processor:** {df.iloc[-1].get('processor', 'N/A')}")
            st.write(f"**Architecture:** {df.iloc[-1].get('architecture', 'N/A')}")
            st.write(f"**Physical Cores:** {df.iloc[-1].get('cpu_cores_physical', 'N/A')}")
            st.write(f"**Logical Cores:** {df.iloc[-1].get('cpu_cores_logical', 'N/A')}")

        with tab8:
            st.subheader("Complete System Snapshot")
            
            # Get the latest system data
            latest_data = df.iloc[-1].to_dict()
            
            # Create a comprehensive list of all properties
            snapshot_data = []

            # Function to recursively add nested items
            # This function will handle nested dictionaries and lists
            def add_nested_items(category, data, prefix=""):
                try:
                    if data is None:
                        snapshot_data.append({
                            "Category": category,
                            "Property": prefix.rstrip('.'),
                            "Value": "N/A"
                        })
                        return
                        
                    if isinstance(data, dict):
                        for key, value in data.items():
                            new_prefix = f"{prefix}{key}."
                            add_nested_items(category, value, new_prefix)
                    elif isinstance(data, list):
                        for i, item in enumerate(data):
                            new_prefix = f"{prefix}{i}."
                            add_nested_items(category, item, new_prefix)
                    else:
                        # Handle all other data types
                        if pd.isna(data):
                            display_value = "N/A"
                        elif isinstance(data, (pd.Timestamp, datetime)):
                            display_value = data.isoformat()
                        elif isinstance(data, (np.integer, int, np.floating, float)):
                            display_value = str(data)
                        else:
                            display_value = str(data)
                        
                        snapshot_data.append({
                            "Category": category,
                            "Property": prefix.rstrip('.'),
                            "Value": display_value
                        })
                except Exception as e:
                    snapshot_data.append({
                        "Category": category,
                        "Property": prefix.rstrip('.'),
                        "Value": f"Error: {str(e)}"
                    })

            # Add all system data
            add_nested_items("System", {
                "system_id": latest_data.get('system_id'),
                "os": f"{latest_data.get('os', '')} {latest_data.get('os_version', '')}",
                "architecture": latest_data.get('architecture'),
                "processor": latest_data.get('processor'),
                "cores": {
                    "physical": latest_data.get('cpu_cores_physical'),
                    "logical": latest_data.get('cpu_cores_logical')
                },
                "ram_size_gb": latest_data.get('ram_size_gb')
            })
            
            # Add other categories
            add_nested_items("CPU", latest_data.get('cpu', {}))
            add_nested_items("Memory", latest_data.get('memory', {}))
            add_nested_items("Disk", latest_data.get('disk', {}))
            add_nested_items("Network", latest_data.get('network', {}))
            
            # Handle network interfaces separately - ensure we get a list
            for i, interface in enumerate(latest_data.get('network_interfaces', [])):
                add_nested_items(f"Network Interface {i+1}", interface)
            
            # Handle GPU data - ensure we get a list even if None
            gpu_data = latest_data.get('gpu', []) or []  # Handle None case
            for i, gpu in enumerate(gpu_data):
                add_nested_items(f"GPU {i+1}", gpu)
            
            # Handle processes
            add_nested_items("Processes", latest_data.get('processes', {}))
            
            # Handle system health
            add_nested_items("System Health", latest_data.get('system_health', {}))

            # Create DataFrame with proper types
            snapshot_df = pd.DataFrame(snapshot_data)
            snapshot_df['Value'] = snapshot_df['Value'].astype(str)

            # Display in vertical format with expandable sections
            for category in sorted(snapshot_df['Category'].unique()):
                with st.expander(f"📁 {category}", expanded=False):
                    st.dataframe(
                        snapshot_df[snapshot_df['Category'] == category][['Property', 'Value']],
                        use_container_width=True,
                        hide_index=True,
                        height=min(400, len(snapshot_df[snapshot_df['Category'] == category]) * 35 + 35),
                        column_config={
                            "Property": st.column_config.Column(width="medium"),
                            "Value": st.column_config.Column(width="large")
                        }
                    )

            # Download options
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                st.download_button(
                    label="📥 Download as CSV",
                    data=snapshot_df.to_csv(index=False),
                    file_name=f"system_snapshot_{selected_system}.csv",
                    mime="text/csv",
                    key=f"csv_download_{selected_system}"
                )
            
            with col2:
                # JSON download using pandas' built-in to_json
                json_str = df.iloc[[-1]].to_json(orient='records', date_format='iso')
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_str,
                    file_name=f"system_snapshot_{selected_system}.json",
                    mime="application/json",
                    key=f"json_download_{selected_system}"
                )
    
    else:
        st.warning("No data found for the selected system and date range")
else:
    st.info("Please select a system from the sidebar")

# Footer
st.sidebar.markdown("""
**Dashboard Features:**
- Real-time system metrics
- Historical trend analysis
- Resource utilization alerts
- Export capabilities
""")