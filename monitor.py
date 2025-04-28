import asyncio
import psutil
import platform
import socket
import uuid
import GPUtil
import os  # Added missing import
from uptime import uptime
import datetime
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from logger.logging import logger
from collections import defaultdict
from typing import Dict, List, Optional, Any

load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

class AsyncSystemMonitor:
    """An asynchronous class to monitor system metrics and store them in MongoDB."""
    def __init__(self):
        logger.info("Initializing AsyncSystemMonitor")
        self.system_id = socket.gethostname()
        self.mac_address = self._get_mac_address()
        self.system_info = {}
        self.collection = None
        self.client = None
        self.loop = asyncio.get_event_loop()

    async def initialize(self):
        """Async initialization that can't be done in __init__"""
        self.client = await self._connect_to_mongodb()
        if self.client:
            self.collection = self.client[DB_NAME][COLLECTION_NAME]
        self.system_info = await self._get_system_info()
        logger.info("AsyncSystemMonitor initialization complete")

    def _get_mac_address(self):
        """Gets MAC address more reliably across platforms."""
        logger.debug("Retrieving MAC address")
        try:
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                          for elements in range(0, 2*6, 8)][::-1])
            logger.debug(f"Retrieved MAC address: {mac}")
            return mac
        except Exception as e:
            logger.warning(f"Could not get MAC address: {str(e)}")
            return "unknown"

    async def _connect_to_mongodb(self) -> Optional[AsyncIOMotorClient]:
        """Establishes async connection to MongoDB with proper error handling."""
        logger.info(f"Connecting to MongoDB at {MONGO_URI}")
        try:
            client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=10000)
            await client.admin.command('ping')  # Test connection
            logger.info(f"Successfully connected to MongoDB database: {DB_NAME}, collection: {COLLECTION_NAME}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return None

    async def _get_system_info(self) -> Dict[str, Any]:
        """Gets static system information that doesn't change."""
        logger.info("Collecting system information")
        try:
            # Run synchronous psutil calls in executor
            cpu_count_logical = await self.loop.run_in_executor(None, psutil.cpu_count, True)
            cpu_count_physical = await self.loop.run_in_executor(None, psutil.cpu_count, False)
            virtual_memory = await self.loop.run_in_executor(None, psutil.virtual_memory)

            system_info = {
                "system_id": self.system_id,
                "mac_address": self.mac_address,
                "os": platform.system(),
                "os_version": platform.version(),
                "kernel_version": platform.release(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "cpu_cores_physical": cpu_count_physical,
                "cpu_cores_logical": cpu_count_logical,
                "ram_size_gb": round(virtual_memory.total / (1024 ** 3), 2),
                "disks": await self._get_disk_info(),
                "network_interfaces": await self._get_network_interfaces(),
                "initial_timestamp": datetime.datetime.utcnow()
            }
            logger.debug(f"System info collected: OS={system_info['os']}, CPU cores={system_info['cpu_cores_logical']}, RAM={system_info['ram_size_gb']}GB")
            return system_info
        except Exception as e:
            logger.error(f"Error collecting system info: {str(e)}")
            return {}

    async def _get_disk_info(self) -> List[Dict[str, Any]]:
        """Gets detailed disk partition information."""
        logger.debug("Collecting disk information")
        disks = []
        try:
            partitions = await self.loop.run_in_executor(None, psutil.disk_partitions)
            for partition in partitions:
                try:
                    usage = await self.loop.run_in_executor(None, psutil.disk_usage, partition.mountpoint)
                    disk_info = {
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total_gb": round(usage.total / (1024 ** 3), 2)
                    }
                    disks.append(disk_info)
                    logger.debug(f"Disk found: {partition.device}, {disk_info['total_gb']}GB total")
                except Exception as e:
                    logger.warning(f"Could not get disk info for {partition.mountpoint}: {str(e)}")
                    continue
            logger.info(f"Collected info for {len(disks)} disk partitions")
            return disks
        except Exception as e:
            logger.error(f"Error collecting disk info: {str(e)}")
            return []

    async def _get_network_interfaces(self) -> List[Dict[str, Any]]:
        """Gets network interface information."""
        logger.debug("Collecting network interface information")
        interfaces = []
        try:
            net_if_addrs = await self.loop.run_in_executor(None, psutil.net_if_addrs)
            for name, addrs in net_if_addrs.items():
                interface = {
                    "name": name,
                    "addresses": [addr.address for addr in addrs]
                }
                interfaces.append(interface)
                logger.debug(f"Network interface found: {name} with {len(interface['addresses'])} addresses")
            logger.info(f"Collected info for {len(interfaces)} network interfaces")
            return interfaces
        except Exception as e:
            logger.error(f"Error collecting network interfaces: {str(e)}")
            return []

    async def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Gets detailed CPU metrics."""
        logger.debug("Collecting CPU metrics")
        try:
            # Run synchronous calls in executor
            cpu_times = await self.loop.run_in_executor(None, psutil.cpu_times_percent, 1)
            cpu_percent = await self.loop.run_in_executor(None, psutil.cpu_percent, 1)
            cpu_percent_percpu = await self.loop.run_in_executor(None, psutil.cpu_percent, 1, True)

            cpu_freq = None
            if hasattr(psutil, 'cpu_freq'):
                cpu_freq_info = await self.loop.run_in_executor(None, psutil.cpu_freq)
                cpu_freq = cpu_freq_info.current if hasattr(cpu_freq_info, 'current') else None

            load_avg = (None, None, None)
            if hasattr(psutil, "getloadavg"):
                load_avg = await self.loop.run_in_executor(None, psutil.getloadavg)

            cpu_metrics = {
                "cpu_usage": cpu_percent,
                "cpu_user": cpu_times.user,
                "cpu_system": cpu_times.system,
                "cpu_idle": cpu_times.idle,
                "cpu_iowait": getattr(cpu_times, 'iowait', None),
                "cpu_steal": getattr(cpu_times, 'steal', None),
                "cpu_frequency_mhz": cpu_freq,
                "load_1min": load_avg[0],
                "load_5min": load_avg[1],
                "load_15min": load_avg[2],
                "cpu_cores_usage": cpu_percent_percpu
            }
            logger.debug(f"CPU usage: {cpu_metrics['cpu_usage']}%, Load avg: {load_avg[0]}")
            return cpu_metrics
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {str(e)}")
            return {}

    async def _get_memory_metrics(self) -> Dict[str, Any]:
        """Gets detailed memory metrics with proper attribute checking."""
        logger.debug("Collecting memory metrics")
        try:
            mem = await self.loop.run_in_executor(None, psutil.virtual_memory)
            swap = await self.loop.run_in_executor(None, psutil.swap_memory)
            
            metrics = {
                "memory_usage_percent": mem.percent,
                "memory_used_gb": round(mem.used / (1024 ** 3), 2),
                "memory_available_gb": round(getattr(mem, 'available', mem.free) / (1024 ** 3), 2),
                "swap_usage_percent": swap.percent,
                "swap_used_gb": round(swap.used / (1024 ** 3), 2)
            }
            
            # Add cached memory if available
            if hasattr(mem, 'cached'):
                metrics["memory_cached_gb"] = round(mem.cached / (1024 ** 3), 2)
            
            # Add buffers if available
            if hasattr(mem, 'buffers'):
                metrics["memory_buffers_gb"] = round(mem.buffers / (1024 ** 3), 2)
            
            # Add page faults if available
            if hasattr(mem, 'page_faults'):
                metrics["page_faults"] = mem.page_faults
            
            logger.debug(f"Memory usage: {metrics['memory_usage_percent']}%, Used: {metrics['memory_used_gb']}GB")
            return metrics
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {str(e)}")
            return {}

    async def _get_disk_metrics(self) -> Dict[str, Any]:
        """Gets detailed disk metrics."""
        logger.debug("Collecting disk metrics")
        try:
            disk_io = await self.loop.run_in_executor(None, psutil.disk_io_counters)
            disk_metrics = {
                "partitions": [],
                "total_read_mb": round(disk_io.read_bytes / (1024 ** 2), 2) if disk_io else None,
                "total_write_mb": round(disk_io.write_bytes / (1024 ** 2), 2) if disk_io else None,
                "read_ops": disk_io.read_count if disk_io else None,
                "write_ops": disk_io.write_count if disk_io else None
            }
            
            partitions = await self.loop.run_in_executor(None, psutil.disk_partitions)
            for partition in partitions:
                try:
                    usage = await self.loop.run_in_executor(None, psutil.disk_usage, partition.mountpoint)
                    partition_info = {
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "usage_percent": usage.percent,
                        "used_gb": round(usage.used / (1024 ** 3), 2),
                        "free_gb": round(usage.free / (1024 ** 3), 2)
                    }
                    disk_metrics["partitions"].append(partition_info)
                    logger.debug(f"Disk partition {partition.device}: {partition_info['usage_percent']}% used")
                except Exception as e:
                    logger.warning(f"Could not get disk usage for {partition.mountpoint}: {str(e)}")
                    continue
            
            logger.debug(f"Disk I/O: Read {disk_metrics['total_read_mb']}MB, Write {disk_metrics['total_write_mb']}MB")
            return disk_metrics
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {str(e)}")
            return {"partitions": []}

    async def _get_network_metrics(self) -> Dict[str, Any]:
        """Gets detailed network metrics."""
        logger.debug("Collecting network metrics")
        try:
            net_io = await self.loop.run_in_executor(None, psutil.net_io_counters)
            connections = await self.loop.run_in_executor(None, psutil.net_connections, 'inet')
            tcp_states = await self._get_tcp_connection_states()
            
            network_metrics = {
                "bytes_sent_mb": round(net_io.bytes_sent / (1024 ** 2), 2),
                "bytes_recv_mb": round(net_io.bytes_recv / (1024 ** 2), 2),
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errors_in": net_io.errin,
                "errors_out": net_io.errout,
                "active_connections": len(connections),
                "tcp_states": tcp_states
            }
            
            logger.debug(f"Network traffic: Sent {network_metrics['bytes_sent_mb']}MB, Received {network_metrics['bytes_recv_mb']}MB")
            logger.debug(f"Active connections: {network_metrics['active_connections']}")
            return network_metrics
        except Exception as e:
            logger.error(f"Error collecting network metrics: {str(e)}")
            return {}

    async def _get_tcp_connection_states(self) -> Dict[str, int]:
        """Counts TCP connections by state."""
        logger.debug("Collecting TCP connection states")
        try:
            states = defaultdict(int)
            connections = await self.loop.run_in_executor(None, psutil.net_connections, 'tcp')
            for conn in connections:
                states[conn.status] += 1
            logger.debug(f"TCP states: {dict(states)}")
            return dict(states)
        except Exception as e:
            logger.error(f"Error collecting TCP states: {str(e)}")
            return {}

    async def _get_gpu_metrics(self) -> Optional[List[Dict[str, Any]]]:
        """Gets GPU metrics if available."""
        logger.debug("Attempting to collect GPU metrics")
        try:
            gpus = await self.loop.run_in_executor(None, GPUtil.getGPUs)
            if not gpus:
                logger.info("No GPUs detected")
                return None
                
            gpu_metrics = [{
                "name": gpu.name,
                "load_percent": gpu.load * 100,
                "memory_usage_percent": gpu.memoryUtil * 100,
                "memory_used_gb": round(gpu.memoryUsed / 1024, 2),
                "memory_total_gb": round(gpu.memoryTotal / 1024, 2),
                "temperature": gpu.temperature,
                "uuid": gpu.uuid
            } for gpu in gpus]
            
            for i, gpu in enumerate(gpu_metrics):
                logger.debug(f"GPU {i+1} ({gpu['name']}): Load {gpu['load_percent']:.1f}%, Memory {gpu['memory_usage_percent']:.1f}%, Temp {gpu['temperature']}°C")
            
            return gpu_metrics
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {str(e)}")
            return None

    async def _get_process_metrics(self) -> Dict[str, Any]:
        """Gets process metrics."""
        logger.debug("Collecting process metrics")
        try:
            processes = []
            proc_iter = await self.loop.run_in_executor(None, psutil.process_iter, 
                                                      ['pid', 'name', 'username', 'cpu_percent', 
                                                       'memory_percent', 'memory_info', 'status'])
            
            for proc in proc_iter:
                try:
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "user": proc.info['username'],
                        "cpu_percent": proc.info['cpu_percent'],
                        "memory_percent": proc.info['memory_percent'],
                        "memory_rss_mb": round(proc.info['memory_info'].rss / (1024 ** 2), 2),
                        "status": proc.info['status']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.debug(f"Could not access process info: {str(e)}")
                    continue
            
            top_cpu = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:10]
            top_mem = sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:10]
            
            zombie_count = len([p for p in processes if p['status'] == psutil.STATUS_ZOMBIE])
            
            process_metrics = {
                "total_processes": len(processes),
                "top_cpu_processes": top_cpu,
                "top_memory_processes": top_mem,
                "zombie_processes": zombie_count
            }
            
            logger.debug(f"Total processes: {process_metrics['total_processes']}, Zombies: {zombie_count}")
            if top_cpu:
                logger.debug(f"Top CPU process: {top_cpu[0]['name']} ({top_cpu[0]['pid']}) at {top_cpu[0]['cpu_percent']:.1f}%")
            if top_mem:
                logger.debug(f"Top memory process: {top_mem[0]['name']} ({top_mem[0]['pid']}) at {top_mem[0]['memory_percent']:.1f}%")
            
            return process_metrics
        except Exception as e:
            logger.error(f"Error collecting process metrics: {str(e)}")
            return {"total_processes": 0, "top_cpu_processes": [], "top_memory_processes": [], "zombie_processes": 0}

    async def _get_system_health(self) -> Dict[str, Any]:
        """Gets system health indicators."""
        logger.debug("Collecting system health metrics")
        try:
            boot_time = datetime.datetime.fromtimestamp(
                await self.loop.run_in_executor(None, psutil.boot_time)
            )
            uptime_secs = uptime()
            uptime_formatted = str(datetime.timedelta(seconds=int(uptime_secs)))
            
            users = await self.loop.run_in_executor(None, psutil.users)
            
            system_health = {
                "uptime_seconds": uptime_secs,
                "uptime_formatted": uptime_formatted,
                "boot_time": boot_time.isoformat(),
                "users": [u.name for u in users],
                "file_descriptors": {
                    "used": psutil.Process().num_fds() if hasattr(psutil.Process(), 'num_fds') else None,
                    "limit": None  # Will be filled differently per OS
                },
                "temperature": await self._get_cpu_temperature()
            }
            
            logger.debug(f"System uptime: {uptime_formatted}, Boot time: {boot_time.isoformat()}")
            logger.debug(f"Active users: {', '.join(system_health['users']) if system_health['users'] else 'None'}")
            return system_health
        except Exception as e:
            logger.error(f"Error collecting system health metrics: {str(e)}")
            return {}

    async def _get_cpu_temperature(self) -> Optional[float]:
        """Gets CPU temperature if available."""
        logger.debug("Trying to get CPU temperature")
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = await self.loop.run_in_executor(None, psutil.sensors_temperatures)
                if temps and 'coretemp' in temps:
                    max_temp = max([t.current for t in temps['coretemp'] if hasattr(t, 'current')])
                    logger.debug(f"CPU temperature: {max_temp}°C")
                    return max_temp
            logger.debug("CPU temperature information not available")
            return None
        except Exception as e:
            logger.error(f"Error getting CPU temperature: {str(e)}")
            return None

    async def collect_metrics(self):
        """Collects all metrics and stores them in MongoDB."""
        logger.info("Starting metrics collection cycle")
        if self.collection is None:  # Check if MongoDB connection is available
            logger.error("No MongoDB connection available, skipping metrics collection")
            return
            
        timestamp = datetime.datetime.utcnow()
        logger.debug(f"Collection timestamp: {timestamp.isoformat()}")
        
        try:
            # Run all metric collection in parallel
            cpu_metrics, memory_metrics, disk_metrics, network_metrics, gpu_metrics, process_metrics, system_health = await asyncio.gather(
                self._get_cpu_metrics(),
                self._get_memory_metrics(),
                self._get_disk_metrics(),
                self._get_network_metrics(),
                self._get_gpu_metrics(),
                self._get_process_metrics(),
                self._get_system_health()
            )

            metrics = {
                "timestamp": timestamp,
                **self.system_info,
                "cpu": cpu_metrics,
                "memory": memory_metrics,
                "disk": disk_metrics,
                "network": network_metrics,
                "gpu": gpu_metrics,
                "processes": process_metrics,
                "system_health": system_health
            }
            
            logger.debug("All metrics collected successfully")
            
            try: 
                result = await self.collection.insert_one(metrics)
                logger.info(f"Metrics logged at {timestamp.isoformat()}, document ID: {result.inserted_id}")
                print(f"Metrics logged at {timestamp.isoformat()}")
            except Exception as e:
                logger.error(f"Failed to store metrics in MongoDB: {str(e)}")
                print(f"Failed to store metrics: {str(e)}")
        except Exception as e:
            logger.error(f"Error during metrics collection: {str(e)}")
            print(f"Error collecting metrics: {str(e)}")

async def main():
    """Main async function to run the monitor."""
    logger.info("Starting AsyncSystemMonitor application")
    monitor = AsyncSystemMonitor()
    await monitor.initialize()  # Added explicit await for initialization
    await monitor.collect_metrics()  # Run once only, no interval loop
    logger.info("AsyncSystemMonitor application completed")

if __name__ == "__main__":
    asyncio.run(main())