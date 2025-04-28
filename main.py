import psutil
import joblib
import pandas as pd
import time
import numpy as np
import subprocess
from datetime import datetime, timedelta
import GPUtil
import os
import sys
from  logger.logging import logger


# Load the trained model
try:
    model = joblib.load('best_model_ml_new.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.critical(f"Failed to load model: {e}")
    sys.exit(1)

# Variables for dynamic enhancement
actual_value = None  # in seconds
next_snapshot_time = None
monitor_process = None
last_snapshot_time = None
cooldown_period = timedelta(seconds=300)  # 5 minutes in seconds (minimum time between snapshots)

def capture_system_metrics():
    """Capture system metrics with exactly the features the model expects."""
    try:
        # CPU Metrics
        cpu_times = psutil.cpu_times_percent(interval=1)
        metrics = {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'cpu_user': cpu_times.user,
            'cpu_system': cpu_times.system,
            'cpu_idle': cpu_times.idle,
            
            # Memory Metrics
            'memory_usage_percent': psutil.virtual_memory().percent,
            'memory_used_gb': round(psutil.virtual_memory().used / (1024 ** 3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024 ** 3), 2),
            'swap_usage_percent': psutil.swap_memory().percent,
            
            # Network Metrics
            'network_bytes_sent_mb': round(psutil.net_io_counters().bytes_sent / (1024 * 1024), 2),
            'network_bytes_recv_mb': round(psutil.net_io_counters().bytes_recv / (1024 * 1024), 2),
            'active_connections': len(psutil.net_connections(kind='inet'))
        }

        # GPU Metrics (try-except block in case GPU isn't available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics.update({
                    'gpu_load_percent': round(gpu.load * 100, 2),
                    'gpu_memory_usage_percent': round(gpu.memoryUtil * 100, 2)
                })
            else:
                metrics.update({
                    'gpu_load_percent': 0,
                    'gpu_memory_usage_percent': 0
                })
        except Exception as gpu_err:
            logger.warning(f"GPU monitoring error: {gpu_err}")
            metrics.update({
                'gpu_load_percent': 0,
                'gpu_memory_usage_percent': 0
            })

        # Create DataFrame with the exact feature order
        input_df = pd.DataFrame([metrics], columns=[
            'cpu_usage', 'cpu_user', 'cpu_system', 'cpu_idle',
            'memory_usage_percent', 'memory_used_gb', 'memory_available_gb', 'swap_usage_percent',
            'gpu_load_percent', 'gpu_memory_usage_percent',
            'network_bytes_sent_mb', 'network_bytes_recv_mb', 'active_connections'
        ])
        
        logger.debug(f"System metrics captured. Shape: {input_df.shape}")
        return input_df

    except Exception as e:
        logger.error(f"Error capturing metrics: {e}")
        raise

def predict_system_performance():
    """Predict system performance and manage snapshot scheduling."""
    global actual_value, next_snapshot_time, monitor_process, last_snapshot_time

    try:
        current_time = datetime.now()
        
        # Check cooldown period
        if last_snapshot_time and (current_time - last_snapshot_time) < cooldown_period:
            remaining_cooldown = (last_snapshot_time + cooldown_period - current_time).total_seconds()
            logger.info(f"In cooldown period. Next snapshot available in: {remaining_cooldown:.1f} seconds")
            return

        input_df = capture_system_metrics()
        
        # Make prediction (model outputs minutes, we convert to seconds)
        prediction = model.predict(input_df)
        predicted_minutes = prediction[0]
        predicted_seconds = predicted_minutes * 60  # Convert to seconds
        logger.info(f"Predicted snapshot interval: {predicted_minutes:.2f} minutes ({predicted_seconds:.0f} seconds)")

        # Initialize or update schedule
        if actual_value is None:
            actual_value = predicted_seconds
            next_snapshot_time = current_time + timedelta(seconds=actual_value)
            logger.info(f"Initial snapshot scheduled at: {next_snapshot_time}")
        else:
            # Only update if prediction is significantly better (20% improvement)
            if predicted_seconds < actual_value * 0.8:
                time_remaining = (next_snapshot_time - current_time).total_seconds()
                
                if time_remaining > max(300, predicted_seconds * 0.2):  # At least 5 minutes or 20% of new interval
                    actual_value = predicted_seconds
                    next_snapshot_time = current_time + timedelta(seconds=actual_value)
                    logger.info(f"Updated snapshot schedule to: {next_snapshot_time}")
                else:
                    logger.info("Maintaining current schedule (too close to snapshot time)")
            else:
                logger.info(f"Prediction {predicted_seconds:.0f}s not better than current {actual_value:.0f}s")

        # Check if it's time to take a snapshot
        if current_time >= next_snapshot_time:
            if take_system_snapshot():
                last_snapshot_time = current_time
                # Schedule next snapshot using current prediction
                actual_value = predicted_seconds
                next_snapshot_time = current_time + timedelta(seconds=actual_value)
                logger.info(f"Snapshot successful. Next scheduled in: {actual_value:.0f} seconds")

    except Exception as e:
        logger.error(f"Error in prediction cycle: {e}")

def take_system_snapshot():
    """Execute the monitor.py script to collect detailed system info."""
    global monitor_process
    
    logger.info("Initiating system snapshot...")
    try:
        # Check if previous process is still running
        if monitor_process and monitor_process.poll() is None:
            logger.warning("Previous snapshot process still running")
            return False
        
        # Start new monitoring process with improved handling
        monitor_process = subprocess.Popen(
            [sys.executable, 'monitor.py'],  # Use current Python interpreter
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        logger.info(f"System snapshot process started with PID: {monitor_process.pid}")
        
        # Capture output without threads
        start_time = time.time()
        while time.time() - start_time < 5:  # Monitor for 5 seconds
            # Check if process completed
            if monitor_process.poll() is not None:
                break
                
            # Read output lines
            stdout_line = monitor_process.stdout.readline()
            if stdout_line:
                logger.info(f"[monitor stdout] {stdout_line.strip()}")
                
            stderr_line = monitor_process.stderr.readline()
            if stderr_line:
                logger.info(f"[monitor stderr] {stderr_line.strip()}")
            
            time.sleep(0.1)  # Prevent busy waiting
        
        # If process is still running after monitoring period
        if monitor_process.poll() is None:
            logger.info("Monitor process is running in background")
        else:
            # Read any remaining output
            for line in monitor_process.stdout:
                logger.info(f"[monitor stdout] {line.strip()}")
            for line in monitor_process.stderr:
                logger.info(f"[monitor stderr] {line.strip()}")
            
            logger.info(f"Monitor process completed with return code: {monitor_process.returncode}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initiate system snapshot: {e}")
        return False

def cleanup():
    """Cleanup resources before exiting."""
    global monitor_process
    try:
        if monitor_process:
            if monitor_process.poll() is None:
                logger.info("Terminating monitor process...")
                monitor_process.terminate()
                try:
                    monitor_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Force killing monitor process...")
                    monitor_process.kill()
                    monitor_process.wait()
            
            # Close file descriptors
            monitor_process.stdout.close()
            monitor_process.stderr.close()
            
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        logger.info("Starting System Performance Monitor")
        
        # Main monitoring loop
        while True:
            predict_system_performance()
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
    finally:
        cleanup()