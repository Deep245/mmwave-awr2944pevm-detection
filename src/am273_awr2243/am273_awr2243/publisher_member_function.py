# Radar Node file, setup to provide highest FPS possible
# It is based on the https://github.com/nhma20/iwr6843aop_pub/tree/main
# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com


#!/usr/bin/env python3
import os
import time
import numpy as np
from threading import Thread, Lock
import serial
import struct
import signal
from datetime import datetime
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
shut_down = 0
data_port = '/dev/ttyACM2'
cli_port = '/dev/ttyACM1'
got_args = False
username = os.path.expanduser('~')
cfg_path = '/mavros_ws/src/am273_awr2243/am273_awr2243/cfg_files/rcs01_range_35m_profile_2026_03_30T10_15_00.cfg'

class TI:
    def __init__(self, sdk_version=4.7,  cli_baud=115200,data_baud=3125000, num_rx=4,num_tx=4,
                 verbose=False, connect=True, mode=0,cli_loc="",data_loc="", cfg_path=""):
        super(TI, self).__init__()
        self.connected = False
        self.verbose = verbose
        self.mode = mode
        self.cfg_path = cfg_path
        if connect:
            self.cli_port = serial.Serial(cli_loc, cli_baud)
            self.data_port = serial.Serial(data_loc, data_baud)
            self.connected = True
        self.sdk_version = sdk_version
        self.num_rx_ant = num_rx
        self.num_tx_ant = num_tx
        self.num_virtual_ant = num_rx * num_tx
        self.frame_times = deque()
        if mode == 0:
            self._initialize()

    def _configure_radar(self, config):
        for i in config:
            self.cli_port.write((i + '\n').encode())
            print(i)
            idx = i.find('frameCfg')
            if idx != -1:
                global ms_per_frame
                ms_per_frame = float(i.split()[6])
                print("Found frameCfg, milliseconds per frame is ", i.split()[6])
            time.sleep(0.01)

    global cfg_path

    def _initialize(self):
        config = [line.rstrip('\r\n') for line in open(self.cfg_path)]
        if self.connected:
            self._configure_radar(config)

        self.config_params = {}

        for i in config:
            split_words = i.split(" ")
            num_rx_ant = 4
            num_tx_ant = 3
            if "profileCfg" in split_words[0]:
                start_freq = int(split_words[2])
                idle_time = int(split_words[3])
                ramp_end_time = float(split_words[5])
                freq_slope_const = int(split_words[8])
                num_adc_samples = int(split_words[10])
                num_adc_samples_round_to2 = 1
                while num_adc_samples > num_adc_samples_round_to2:
                    num_adc_samples_round_to2 *= 2
                dig_out_sample_rate = int(split_words[11])
            elif "frameCfg" in split_words[0]:
                chirp_start_idx = int(split_words[1])
                chirp_end_idx = int(split_words[2])
                num_loops = int(split_words[3])
                num_frames = int(split_words[5])
                frame_periodicity = float(split_words[6])

        num_chirps_per_frame = (chirp_end_idx - chirp_start_idx + 1) * num_loops
        self.config_params["numDopplerBins"] = num_chirps_per_frame / num_tx_ant
        self.config_params["numRangeBins"] = num_adc_samples_round_to2
        self.config_params["rangeResolutionMeters"] = (3e8 * dig_out_sample_rate * 1e3) / (
            2 * freq_slope_const * 1e12 * num_adc_samples)
        self.config_params["rangeIdxToMeters"] = (3e8 * dig_out_sample_rate * 1e3) / (
            2 * freq_slope_const * 1e12 * self.config_params["numRangeBins"])
        self.config_params["dopplerResolutionMps"] = 3e8 / (
            2 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 *
            self.config_params["numDopplerBins"] * num_tx_ant)
        self.config_params["maxRange"] = (300 * 0.9 * dig_out_sample_rate) / (2 * freq_slope_const * 1e3)
        self.config_params["maxVelocity"] = 3e8 / (
            4 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 * num_tx_ant)

    def close(self):
        print("Shutting down sensor")
        self.cli_port.write('sensorStop\n'.encode())
        self.cli_port.close()
        self.data_port.close()

    def _read_buffer(self):
        byte_buffer = self.data_port.read(self.data_port.in_waiting)
        if self.data_port.in_waiting > 4096:
            print("[WARN] Serial backlog detected")
        return byte_buffer

    def _parse_header_data(self, byte_buffer, idx):
        magic, idx = self._unpack(byte_buffer, idx, order='>', items=1, form='Q')
        (version, length, platform, frame_num, cpu_cycles, num_obj, num_tlvs), idx = self._unpack(byte_buffer, idx, items=7, form='I')
        subframe_num, idx = self._unpack(byte_buffer, idx, items=1, form='I')
        return (version, length, platform, frame_num, cpu_cycles, num_obj, num_tlvs, subframe_num), idx

    def _parse_header_tlv(self, byte_buffer, idx):
        (tlv_type, tlv_length), idx = self._unpack(byte_buffer, idx, items=2, form='I')
        return (tlv_type, tlv_length), idx

    def _parse_msg_detected_points(self, byte_buffer, idx):
        (x, y, z, vel), idx = self._unpack(byte_buffer, idx, items=4, form='f')
        return (x, y, z, vel), idx

    def _process_detected_points(self, byte_buffer):
        idx = byte_buffer.index(MAGIC_WORD)
        header_data, idx = self._parse_header_data(byte_buffer, idx)
        num_tlvs = header_data[6]
        frame_num = header_data[3]
        now = time.time()
        if not hasattr(self, 'last_frame_num'):
            self.last_frame_num = frame_num
        if not hasattr(self, 'last_print_time'):
            self.last_print_time = now
        self.frame_times.append((now, frame_num))
        while self.frame_times and (now - self.frame_times[0][0]) > 1.0:
            self.frame_times.popleft()
        if now - self.last_print_time >= 1.0:
            if self.frame_times:
                first_frame = self.frame_times[0][1]
                last_frame = self.frame_times[-1][1]
                fps = (last_frame - first_frame) & 0xFFFFFFFF
                print(f"[FPS]: {fps}")
            self.last_print_time = now
        (tlv_type, tlv_length), idx = self._parse_header_tlv(byte_buffer, idx)
        num_points = int(tlv_length / 16)
        data = np.zeros((num_points, 4), dtype=np.float64)
        for i in range(num_points):
            (x, y, z, vel), idx = self._parse_msg_detected_points(byte_buffer, idx)
            data[i][0] = x
            data[i][1] = y
            data[i][2] = z
            data[i][3] = vel  # radial velocity
        return data

    @staticmethod
    def _unpack(byte_buffer, idx, order='', items=1, form='I'):
        size = {'H': 2, 'h': 2, 'I': 4, 'Q': 8, 'f': 4}
        try:
            data = struct.unpack(order + str(items) + form, byte_buffer[idx:idx + (items * size[form])])
            if len(data) == 1:
                data = data[0]
            return data, idx + (items * size[form])
        except:
            return None

class Detected_Points:
    def __init__(self, cli_loc=cli_port, data_loc=data_port, cfg_path=cfg_path):
        self.MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        self.ti = TI(cli_loc=cli_loc, data_loc=data_loc, cfg_path=cfg_path)
        self.interval = 0.001
        self.data = b''
        self.warn = 0

    def data_stream_iterator(self):
        while 1:
            time.sleep(self.interval)
            byte_buffer = self.ti._read_buffer()
            if len(byte_buffer) == 0:
                self.warn += 1
            else:
                self.warn = 0
            if self.warn > 100:
                print("Wrong")
                break
            self.data += byte_buffer
            try:
                idx1 = self.data.index(MAGIC_WORD)
                idx2 = self.data.index(MAGIC_WORD, idx1 + 1)
            except ValueError:
                if len(self.data) > 2**14:
                    self.data = self.data[-8192:]
                continue
            try:
                points = self.ti._process_detected_points(self.data)
            except Exception as e:
                print(f"[WARN] Frame parse failed: {e}")
                self.data = self.data[-8192:]
                continue
            self.data = self.data[idx2:]
            yield points
            global shut_down
            if shut_down == 1:
                break
        self.ti.close()

from threading import Thread, Lock
import signal
import time
import numpy as np
import os
from datetime import datetime

shut_down = 0
cli_port = '/dev/ttyACM1'
data_port = '/dev/ttyACM2'
cfg_path = os.path.expanduser('~/mavros_ws/src/am273_awr2243/am273_awr2243/cfg_files/rcs01_range_35m_profile_2026_03_30T10_15_00.cfg')

class awr2243_interface(Node):
    def __init__(self):
        super().__init__('awr2243_node')
        self.publisher_ = self.create_publisher(PointCloud2, 'radarFrame', 10)
        self.stream = None

    def get_data(self):
        detected_points = Detected_Points(cli_port, data_port, cfg_path)
        max_vel = detected_points.ti.config_params.get("maxVelocity", 7.06)
        alias_thresh = max_vel * 0.90  # points with |vel| above this are Doppler-aliased static clutter
        print(f"[RADAR] maxVelocity={max_vel:.4f} m/s, alias filter threshold={alias_thresh:.4f} m/s")
        self.stream = detected_points.data_stream_iterator()
        try:
            for data in self.stream:
                # Filter Doppler-aliased static clutter: zero out velocities near ±maxVelocity
                alias_mask = np.abs(data[:, 3]) >= alias_thresh
                data[alias_mask, 3] = 0.0
                vels = data[:, 3]
                print(f"[RADAR] pts={len(vels)} vel={np.round(vels, 4).tolist()}")
                pc2_msg = self.create_pointcloud2_msg(data)
                self.publisher_.publish(pc2_msg)
                if shut_down:
                    break
        except Exception as e:
            print("Data stream error:", e)
        finally:
            detected_points.ti.close()

    def create_pointcloud2_msg(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "radar"
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='velocity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        data = []
        for p in points:
            data.append(struct.pack('ffff', p[0], p[1], p[2], p[3]))
        pc2 = PointCloud2()
        pc2.header = header
        pc2.height = 1
        pc2.width = len(points)
        pc2.is_dense = True
        pc2.is_bigendian = False
        pc2.fields = fields
        pc2.point_step = 16
        pc2.row_step = 16 * len(points)
        pc2.data = b''.join(data)
        return pc2

def ctrlc_handler(signum, frame):
    global shut_down
    shut_down = 1
    print("Exiting...")

def main(args=None):
    global shut_down
    signal.signal(signal.SIGINT, ctrlc_handler)
    rclpy.init(args=args)
    radar = awr2243_interface()
    data_thread = Thread(target=radar.get_data, daemon=True)
    data_thread.start()
    try:
        rclpy.spin(radar)
    except KeyboardInterrupt:
        shut_down = 1
    finally:
        data_thread.join()
        radar.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

