# myorawfast.py
# Author: Charles Lambelet
# Created: October 2019

import arcade
import sys
import enum
import re
import struct
import serial
import time
import numpy as np
import pandas as pd
from serial.tools.list_ports import comports
from periodicthread import PeriodicThread

### feedback game
# from myo_feedback import *

### collecting coins
from myo_sprite import *

import arcade
from pickle import load
import glob2
from scipy import stats
from itertools import islice

'''
#########
# NOTES #
#########
Handlers are useful to pass variables across classes
when these variables are regularily updated!
'''

CHANNELS = 8

# Function that calculates the similarity index (cosine similarity) The reference is a 1x8 vector, which contain RMS
# of each channel (c1 - c8). Reference was Hao-Ping's right hand (gestureProfile.pkl)
def similarity(A_vector, reference):
    A_vector = np.sqrt(np.mean(A_vector**2, axis = 0))
    reference = np.array(reference)
    C = np.multiply(A_vector, reference).sum() / (np.sqrt((A_vector**2).sum() * (reference**2).sum()))
    
    return C

def pack(fmt, *args):
    return struct.pack('<' + fmt, *args)


def unpack(fmt, *args):
    return struct.unpack('<' + fmt, *args)


def multichr(ords):
    """convert list to bytes"""
    if sys.version_info[0] >= 3:
        return bytes(ords)
    else:
        return ''.join(map(chr, ords))


def multiord(b):
    """convert bytes to list"""
    if sys.version_info[0] >= 3:
        return list(b)
    else:
        return map(ord, b)
    
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result    


class Arm(enum.Enum):
    UNKNOWN = 0
    RIGHT = 1
    LEFT = 2


class XDirection(enum.Enum):
    UNKNOWN = 0
    X_TOWARD_WRIST = 1
    X_TOWARD_ELBOW = 2


class Pose(enum.Enum):
    REST = 0
    FIST = 1
    WAVE_IN = 2
    WAVE_OUT = 3
    FINGERS_SPREAD = 4
    THUMB_TO_PINKY = 5
    UNKNOWN = 255


class Packet(object):
    def __init__(self, ords):
        self.typ = ords[0]
        self.cls = ords[2]
        self.cmd = ords[3]
        self.payload = multichr(ords[4:])

    def __repr__(self):
        return 'Packet(%02X, %02X, %02X, [%s])' % \
            (self.typ, self.cls, self.cmd,
             ' '.join('%02X' % b for b in multiord(self.payload)))


class BT(object):
    """Implements the non-Myo-specific details of the Bluetooth protocol."""
    def __init__(self, tty):
        self.ser = serial.Serial(port=tty, baudrate=115200, dsrdtr=1, timeout=0.01)
        self.buf = []
        self.handlers = []

    # internal data-handling method to receive bytes from serial
    def recv_packet(self):

        # while number of bytes in input buffer of serial > 150
        while self.ser.inWaiting() > 150:
            self.ser.reset_input_buffer()

        # if there are bytes in input buffer of serial to be collected
        if self.ser.inWaiting() > 0:
            while True:
                # collect bytes
                c = self.ser.read()
                if not c:
                    return None
                ret = self.proc_byte(ord(c))  # ord() returns unicode number
                if ret:
                    if ret.typ == 0x80:
                        # execute handle_data() from MyoRaw class which actually converts the received bytes into sEMG values
                        self.handle_event(ret)
                    return ret

    # internal data-handling method to process bytes coming from serial
    def proc_byte(self, c):
        if not self.buf:
            if c in [0x00, 0x80, 0x08, 0x88]:  # [BLE response pkt, BLE event pkt, wifi response pkt, wifi event pkt]
                self.buf.append(c)
            return None
        elif len(self.buf) == 1:
            self.buf.append(c)
            self.packet_len = 4 + (self.buf[0] & 0x07) + self.buf[1]
            return None
        else:
            self.buf.append(c)

        if self.packet_len and len(self.buf) == self.packet_len:
            p = Packet(self.buf)  # create packet p
            self.buf = []
            return p
        return None

    # in our case there is actually only 1 function in the handler which is handle_data() from MyoRaw
    def handle_event(self, p):
        for h in self.handlers:
            h(p)

    # add functions to the handler
    def add_handler(self, h):
        self.handlers.append(h)

    # remove functions from the handler
    def remove_handler(self, h):
        try:
            self.handlers.remove(h)
        except ValueError:
            pass

    def wait_event(self, cls, cmd):
        res = [None]

        # how does it work? where does p come from?
        def h(p):
            if p.cls == cls and p.cmd == cmd:
                res[0] = p
        self.add_handler(h)
        while res[0] is None:
            self.recv_packet()
        self.remove_handler(h)
        return res[0]

    # specific BLE commands
    def connect(self, addr):
        return self.send_command(6, 3, pack('6sBHHHH', multichr(addr), 0, 6, 6, 64, 0))

    def get_connections(self):
        return self.send_command(0, 6)

    def discover(self):
        return self.send_command(6, 2, b'\x01')

    def end_scan(self):
        return self.send_command(6, 4)

    def disconnect(self, h):
        return self.send_command(3, 0, pack('B', h))

    def read_attr(self, con, attr):
        self.send_command(4, 4, pack('BH', con, attr))
        return self.wait_event(4, 5)

    def write_attr(self, con, attr, val):
        self.send_command(4, 5, pack('BHB', con, attr, len(val)) + val)
        return self.wait_event(4, 1)

    def send_command(self, cls, cmd, payload=b'', wait_resp=True):
        s = pack('4B', 0, len(payload), cls, cmd) + payload
        self.ser.write(s)

        while True:
            p = self.recv_packet()
            if p is None:
                continue
            elif p.typ == 0:
                return p
            else:
                self.handle_event(p)


class MyoRaw(object):
    """Implements the Myo-specific communication protocol."""
    def __init__(self, tty=None):
        if tty is None:
            tty = self.detect_tty()
        if tty is None:
            raise ValueError('Myo dongle not found!')

        self.bt = BT(tty)
        self.conn = None
        self.emg_handlers = []  # this array actually stores functions!
        self.imu_handlers = []
        self.arm_handlers = []
        self.pose_handlers = []
        self.battery_handlers = []
        self.emg_list = []
        self.up_pressed = False
        self.down_pressed = False  
        self.right_pressed = False
        self.left_pressed = False
        self.MOVEMENT_SPEED = 3
        self.color = arcade.color.WHITE
        self.similarity = 0
        self.magnitude = 0
        self.model = None
        self.label = 'Rest'
        self.label_list = []
        
        self.gestureProfile = pd.read_pickle('gestureProfile.pkl')
        self.gestureProfile = self.gestureProfile.set_index('Gesture')


    def detect_tty(self):
        for p in comports():
            if re.search(r'PID=2458:0*1', p[2]):
                print('using device:', p[0])
                return p[0]

        return None

    def run(self):
        self.bt.recv_packet()

    def connect(self):
        # stop everything from before
        self.bt.end_scan()
        self.bt.disconnect(0)
        self.bt.disconnect(1)
        self.bt.disconnect(2)

        # start scanning
        print('scanning for Myo...')
        self.bt.discover()
        while True:
            p = self.bt.recv_packet()
            if p:
                if p.payload.endswith(b'\x06\x42\x48\x12\x4A\x7F\x2C\x48\x47\xB9\xDE\x04\xA9\x01\x00\x06\xD5'):
                    self.addr = list(multiord(p.payload[2:8]))
                    break

        self.bt.end_scan()

        # connect and wait for status event
        conn_pkt = self.bt.connect(self.addr)
        self.conn = multiord(conn_pkt.payload)[-1]
        self.bt.wait_event(3, 0)

        print('Myo connected')

        # get firmware version
        fw = self.read_attr(0x17)
        _, _, _, _, v0, v1, v2, v3 = unpack('BHBBHHHH', fw.payload)
        print('firmware version: %d.%d.%d.%d' % (v0, v1, v2, v3))

        self.old = (v0 == 0)
        if self.old:
            print('FIRWARE ERROR: update myo firmware to 1.5.1970.2')

        # enable IMU data
        # self.write_attr(0x1d, b'\x01\x00')  # enable
        self.write_attr(0x1d, b'\x00\x00')  # disable
        # enable on/off arm notifications
        # self.write_attr(0x24, b'\x02\x00')  # enable
        self.write_attr(0x24, b'\x00\x00')  # disable
        # enable EMG notifications
        self.start_raw()
        # enable battery notifications
        self.write_attr(0x12, b'\x01\x10')

        # handler function which is called periodically to collect emg data
        def handle_data(p):
            if (p.cls, p.cmd) != (4, 5):
                return

            # get the attribute of the packet
            c, attr, typ = unpack('BHB', p.payload[:4])
            # extract the payload from the packet
            pay = p.payload[5:]

            # read notification handles corresponding to the for EMG characteristics
            if attr == 0x2b or attr == 0x2e or attr == 0x31 or attr == 0x34:
                '''According to http://developerblog.myo.com/myocraft-emg-in-the-bluetooth-protocol/
                each characteristic sends two secuential readings in each update,
                so the received payload is split in two samples. According to the
                Myo BLE specification, the data type of the EMG samples is int8_t.
                '''
                emg1 = struct.unpack('<8b', pay[:8])  # unpack emg data 1 from payload
                emg2 = struct.unpack('<8b', pay[8:])  # unpack emg data 2 from payload
#                 print(emg1)
#                 print(emg2)
                self.on_emg(emg1)  # collect emg data 1
                self.on_emg(emg2)  # collect emg data 2

            # read IMU characteristic handle
            elif attr == 0x1c:
                vals = unpack('10h', pay)
                quat = vals[:4]
                acc = vals[4:7]
                gyro = vals[7:10]
                self.on_imu(quat, acc, gyro)

            # read classifier characteristic handle
            elif attr == 0x23:
                typ, val, xdir, _, _, _ = unpack('6B', pay)
                if typ == 1:  # on arm
                    self.on_arm(Arm(val), XDirection(xdir))
                elif typ == 2:  # removed from arm
                    self.on_arm(Arm.UNKNOWN, XDirection.UNKNOWN)
                elif typ == 3:  # pose
                    self.on_pose(Pose(val))

            # read battery characteristic handle
            elif attr == 0x11:
                battery_level = ord(pay)
                #print('battery level: %d' % battery_level)
                self.on_battery(battery_level)

            else:
                print('data with unknown attr: %02X %s' % (attr, p))
            self.emg_list.append(emg1)
            self.emg_list.append(emg2)
            
            # Use pretrained tensorflow model to predict the gesture from myo and control the sprite
            emg_data = []
            majorityVoteWindowSize = 45
            if len(self.emg_list) >= 400:

                emg_data = self.emg_list[-30:]
                emg_data = np.array(emg_data).reshape((-1, 30, 8, 1))
                temp_label = np.argmax(self.model.predict(emg_data, batch_size=1), axis = -1)[0]
                self.label_list.append(temp_label)

                if len(self.label_list) >= majorityVoteWindowSize:

                    # take the majority vote from multiple frames
                    self.label = stats.mode(self.label_list[-majorityVoteWindowSize:])[0][0]
                                                                      
                    if np.mean(np.sqrt(np.sum(np.array(self.emg_list[-5:])**2, axis = 0))) <= 10 or self.label == 3:
                        
                        self.label = 'Rest'
                        self.similarity = 0
                        print('Rest')
                        self.similarity = 0
                        self.magnitude = 0
                        self.up_pressed = False
                        self.down_pressed = False
                        self.left_pressed = False
                        self.right_pressed = False
                        self.color  = arcade.color.WHITE
                        
                        
                    elif self.label == 0:
                        self.label = 'Finger mass extension'
                        emg_temp = np.array(self.emg_list[-majorityVoteWindowSize:]).reshape((-1, 8))
                        
                        # Bar
                        self.similarity = similarity(emg_temp, self.gestureProfile.loc[self.label][0])
                        self.magnitude = np.mean(np.sqrt(np.mean(emg_temp**2, axis = 0))) / np.mean(self.gestureProfile.loc[self.label][0])

                        print('Finger mass extension')
                        self.up_pressed = False
                        self.down_pressed = False
                        self.left_pressed = False
                        self.right_pressed = True  
                        
                        if float(np.abs(self.similarity)) >= 0.90:
                            self.color  = arcade.color.GREEN
                        elif float(np.abs(self.similarity)) >= 0.85:
                            self.color = arcade.color.YELLOW
                        else:
                            self.color = arcade.color.RED
                        
                    elif self.label == 1:
                        self.label = 'Fist'
                        emg_temp = np.array(self.emg_list[-majorityVoteWindowSize:]).reshape((-1, 8))
                        
                        # Bar 
                        self.similarity = similarity(emg_temp, self.gestureProfile.loc[self.label][0])
                        self.magnitude = np.mean(np.sqrt(np.mean(emg_temp**2, axis = 0))) / np.mean(self.gestureProfile.loc[self.label][0])
                        
                        print('Fist')
                        self.up_pressed = False
                        self.down_pressed = True
                        self.left_pressed = False
                        self.right_pressed = False
                        
                        if float(np.abs(self.similarity)) >= 0.90:
                            self.color  = arcade.color.GREEN
                        elif float(np.abs(self.similarity)) >= 0.85:
                            self.color = arcade.color.YELLOW
                        else:
                            self.color = arcade.color.RED
                        
                    elif self.label == 2:
                        self.label = 'Opposition'
                        emg_temp = np.array(self.emg_list[-majorityVoteWindowSize:]).reshape((-1, 8))
                        
                        # Bar 
                        self.similarity = similarity(emg_temp, self.gestureProfile.loc[self.label][0])
                        self.magnitude = np.mean(np.sqrt(np.mean(emg_temp**2, axis = 0))) / np.mean(self.gestureProfile.loc[self.label][0])
                        
                        print('Opposition')
                        self.up_pressed = False
                        self.down_pressed = False 
                        self.left_pressed = True
                        self.right_pressed = False   
                        
                        if float(np.abs(self.similarity)) >= 0.90:
                            self.color  = arcade.color.GREEN
                        elif float(np.abs(self.similarity)) >= 0.85:
                            self.color = arcade.color.YELLOW
                        else:
                            self.color = arcade.color.RED
                            
                    elif self.label == 4:
                        self.label = 'Wrist and finger extension'
                        emg_temp = np.array(self.emg_list[-majorityVoteWindowSize:]).reshape((-1, 8))
                        
                        # Bar 
                        self.similarity = similarity(emg_temp, self.gestureProfile.loc[self.label][0])
                        self.magnitude = np.mean(np.sqrt(np.mean(emg_temp**2, axis = 0))) / np.mean(self.gestureProfile.loc[self.label][0])
                        
                        print('Wrist and finger extension')
                        self.up_pressed = True
                        self.down_pressed = False                      
                        self.left_pressed = False
                        self.right_pressed = False
                        if float(np.abs(self.similarity)) >= 0.90:
                            self.color  = arcade.color.GREEN
                        elif float(np.abs(self.similarity)) >= 0.85:
                            self.color = arcade.color.YELLOW
                        else:
                            self.color = arcade.color.RED
                
                            
        # add the function to the handler to ultimately retrieve the emg values
        self.bt.add_handler(handle_data)

    def reconnect(self):
        # reconnect and wait for status event
        conn_pkt = self.bt.connect(self.addr)
        self.conn = multiord(conn_pkt.payload)[-1]
        self.bt.wait_event(3, 0)

        print('Myo reconnected')

        # enable IMU data
        # self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x1d, b'\x00\x00')
        # enable on/off arm notifications
        # self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x24, b'\x00\x00')
        # enable EMG notifications
        self.start_raw()
        # enable battery notifications
        self.write_attr(0x12, b'\x01\x10')

    def write_attr(self, attr, val):
        if self.conn is not None:
            self.bt.write_attr(self.conn, attr, val)

    def read_attr(self, attr):
        if self.conn is not None:
            return self.bt.read_attr(self.conn, attr)
        return None

    def disconnect(self):
        if self.conn is not None:
            self.bt.disconnect(self.conn)
        print('Myo disconnected')

    def sleep_mode(self, mode):
        self.write_attr(0x19, pack('3B', 9, 1, mode))

    def power_off(self):
        self.write_attr(0x19, b'\x04\x00')

    def start_raw(self):

        ''' To get raw EMG signals, we subscribe to the four EMG notification
        characteristics by writing a 0x0100 command to the corresponding handles.
        '''
        self.write_attr(0x2c, b'\x01\x00')  # Suscribe to EmgData0Characteristic
        self.write_attr(0x2f, b'\x01\x00')  # Suscribe to EmgData1Characteristic
        self.write_attr(0x32, b'\x01\x00')  # Suscribe to EmgData2Characteristic
        self.write_attr(0x35, b'\x01\x00')  # Suscribe to EmgData3Characteristic

        '''Bytes sent to handle 0x19 (command characteristic) have the following
        format: [command, payload_size, EMG mode, IMU mode, classifier mode]
        According to the Myo BLE specification, the commands are:
            0x01 -> set EMG and IMU
            0x03 -> 3 bytes of payload
            0x02 -> send 50Hz filtered signals
            0x01 -> send IMU data streams
            0x01 -> send classifier events
        '''
        # self.write_attr(0x19, b'\x01\x03\x02\x01\x01')
        self.write_attr(0x19, b'\x01\x03\x02\x00\x00')

    def vibrate(self, length):
        if length in range(1, 4):
            # first byte tells it to vibrate; purpose of second byte is unknown (payload size?)
            self.write_attr(0x19, pack('3B', 3, 1, length))  # length in second

    def set_leds(self, logo, line):
        self.write_attr(0x19, pack('8B', 6, 6, *(logo + line)))  # logo and line should be lists of 3 elements => color code

    def add_emg_handler(self, h):
        # append the functions in the handler
        self.emg_handlers.append(h)

    def add_imu_handler(self, h):
        self.imu_handlers.append(h)

    def add_pose_handler(self, h):
        self.pose_handlers.append(h)

    def add_arm_handler(self, h):
        self.arm_handlers.append(h)

    def add_battery_handler(self, h):
        self.battery_handlers.append(h)

    def on_emg(self, emg):
        # for each function in the handler (emg_handlers) pass 1 emg sample!
        # in our case, there is only 1 function in the handler.
        for h in self.emg_handlers:
            h(emg)  # pass 1 emg sample to the function contained within the handler

    def on_imu(self, quat, acc, gyro):
        for h in self.imu_handlers:
            h(quat, acc, gyro)

    def on_pose(self, p):
        for h in self.pose_handlers:
            h(p)

    def on_arm(self, arm, xdir):
        for h in self.arm_handlers:
            h(arm, xdir)

    def on_battery(self, battery_level):
        for h in self.battery_handlers:
            h(battery_level)


class MyoMain():

    """Myo data collection."""
    def __init__(self, model):
        self.window = MyGame()
        self.window.setup()
        
        self.mr = MyoRaw()
        self.mr.add_emg_handler(self.emg_handler)  # IMPORTANT! pass the function emg_handler to access emg data from MyoRaw
        self.mr.add_battery_handler(self.battery_handler)
        self.mr.model = model
        
        self.pt = PeriodicThread(self.collect, 1.0/120.0)  # should be 100Hz but see below for explanations
        self.buffer_size = 10  # FIFO emg buffer size.
        self.emg_buffer = []  # FIFO emg buffer
        self.sample_n = 0  # id number for each data sample once collected
        self.battery_status = 0
        self.emg_buffer_handlers = []  # handlers array used to pass variables (functions) to class DataWidget()

        
    def connect(self):
        self.mr.connect()
        

    def no_sleep(self):
        self.mr.sleep_mode(1)

    def disconnect(self):
        self.mr.disconnect()

    def reconnect(self):
        self.mr.reconnect()

    # start collecting data every 0.01s (100Hz) since Myo sends 2 data samples at a time.
    # at the end, the sampling rate is 0.005s per sample (200Hz).
    # however, to be sure to collect all data, we sample a bit faster as initialized above => 120Hz
    def start_collect(self):
        # start thread at 120Hz
        self.pt.start()
        print("Periodic thread @120Hz for Myo data started")
        arcade.run()
        
            
    # collect all data, i.e. emg, battery, etc...
    def collect(self):
        self.mr.run()

    def stop_collect(self):
        # kill thread at 120Hz
        self.pt.cancel()
        print("Periodic thread @120Hz for Myo data terminated")

    # emg samples are buffered here. 10 samples FIFO buffer seems to be enough to not miss any data
    # since the collection is at 100Hz (i.e. 200Hz) while the displaying is at 60Hz.
    def emg_handler(self, emg, *args):  # *args: if more argument needs to be passed to the function
        """Add sEMG data to 10 samples FIFO buffer."""
        self.emg = emg  # this variable receives and contains 1 data sample at a time, i.e. 8 channel values
        self.ts = time.time()
        if self.emg:
            # if buffer empty, add new emg
            if len(self.emg_buffer) == 0:
                self.emg_buffer = [self.sample_n] + [list(self.emg) + [self.ts]]
            # else if buffer already filled, append new data
            else:
                self.emg_buffer.append([self.sample_n] + list(self.emg) + [self.ts])

            # if buffer exceeds buffer size, remove old emg data that have already been passed for filtering
            if len(self.emg_buffer) > self.buffer_size:
                # delete first data samples (oldest elements) from the list
                del self.emg_buffer[:len(self.emg_buffer) - self.buffer_size]

            # at the end, self.emg_buffer is always 10 data samples long
            # and the subsequent buffers have always one new data sample more and one old data sample less
            # print(self.emg_buffer)

            # pass emg_buffer to the handler to access data in class DataWidget()
            self.on_emg_buffer(self.emg_buffer)

            self.sample_n += 1
        
        
        # sprite setting / mapping
        self.window.up_pressed = self.mr.up_pressed
        self.window.down_pressed = self.mr.down_pressed
        self.window.right_pressed = self.mr.right_pressed
        self.window.left_pressed = self.mr.left_pressed
        self.window.label = self.mr.label
        self.window.similarity = self.mr.similarity
        self.window.magnitude = self.mr.magnitude
        self.window.color = self.mr.color
        
        self.MOVEMENT_SPEED = self.mr.MOVEMENT_SPEED
        
            

    # it seems that battery status are sent by the Myo sporadically or maybe only sent when the status changes
    def battery_handler(self, battery_status, *args):
        self.battery_status = battery_status

    def add_emg_buffer_handler(self, h):
        self.emg_buffer_handlers.append(h)

    def on_emg_buffer(self, emg_buffer):
        for h in self.emg_buffer_handlers:
            h(emg_buffer)