#!/usr/bin/env python
import sys, os
TOP_DIR = os.path.dirname(os.path.abspath(__file__))
NET_PATH = os.path.join(TOP_DIR, '../net')
MODEL_PATH = os.path.join(TOP_DIR, '../net/models')
sys.path.append(NET_PATH)

import signal
import collections
import pyaudio
import time
import wave
import logging
import numpy as np
from scipy.io import loadmat
import pdb
import matplotlib.pyplot as plt
import data as getdata
from predict_spec import detect_online, get_model
import uuid
from scipy.io import wavfile
import ast

interrupted = False

def signal_handler(signal, frame):
    global interrupted
    interrupted = True

def interrupt_callback():
    global interrupted
    return interrupted

logging.basicConfig()
logger = logging.getLogger("detector")
logger.setLevel(logging.INFO)
DETECT_SOUND = os.path.join(TOP_DIR, "data/beep.wav")

class RingBuffer(object):
    """Ring buffer to hold audio from PortAudio"""
    def __init__(self, size = 4096):
        self._buf = collections.deque(maxlen=size)

    def extend(self, data):
        """Adds data to the end of buffer"""
        self._buf.extend(data)

    def get(self):
        """Retrieves data from the buffer"""
        tmp = ''.join(self._buf)
        return tmp


def play_audio_file(fname=DETECT_SOUND):
    """Simple callback function to play a wave file."""
    if True:
        f = wave.open(fname, 'rb')
        wavData = f.readframes(f.getnframes())
        audio = pyaudio.PyAudio()
        stream_out = audio.open(
            format=audio.get_format_from_width(f.getsampwidth()),
            channels=f.getnchannels(),
            rate=f.getframerate(), input=False, output=True)
        stream_out.start_stream()
        stream_out.write(wavData)
        time.sleep(0.2)
        stream_out.stop_stream()
        stream_out.close()
        audio.terminate()

class Detector(object):
    """
    Detect whether a keyword specified by `model_str`
    exists in a microphone input stream.
    """
    def __init__(self, model_str, epoch=None, detTh=1.5,
                 audio_gain=1.):

        def audio_callback(in_data, frame_count, time_info, status):
            self.ring_buffer.extend(in_data)
            play_data = chr(0) * len(in_data)
            return play_data, pyaudio.paContinue

        print('Compiling... This may take a minute.')
        self.model, self.modelType, self.winLen, self.offset, self.scale = get_model(model_str, epoch=epoch)

        self.winLen_s = 1.6
        self.detWait = 10
        self.audioGain = audio_gain
        self.detTh = detTh
        self.prob_prev = np.array([0.])
        self.waitCount = 0
        self.waiting = False

        self.numChannels = 1
        self.sampleRate = 16000
        self.ring_buffer = RingBuffer(2 * self.numChannels * self.sampleRate * self.winLen_s)
        self.audio = pyaudio.PyAudio()
        self.stream_in = self.audio.open(
            input=True, output=False,
            format=self.audio.get_format_from_width(16 / 8),
            channels=1,
            rate=self.sampleRate,
            frames_per_buffer=1600,
            stream_callback=audio_callback)


    def start(self, detected_callback=play_audio_file,
              interrupt_check=lambda: False,
              sleep_time=0.20,debug=False):

        if interrupt_check():
            logger.debug("detect voice return")
            return

        logger.debug("detecting...")

        #print time.time()
        #print self.detTh
        while True:
            if interrupt_check():
                logger.debug("detect voice break")
                break
            data = self.ring_buffer.get()
            time.sleep(sleep_time)

            data_numeric = np.float32(np.frombuffer(data, dtype='<i2')) * self.audioGain

            # If we have enough data in the buffer
            if data_numeric.shape[0] == self.sampleRate * self.winLen_s:
                # Get a detection result
                flag, self.prob_prev, self.waitCount, self.waiting = detect_online(data_numeric,
                        self.prob_prev,
                        self.model, self.modelType, self.winLen,
                        self.offset, self.scale,
                        self.detWait, self.detTh,
                        self.waitCount, self.waiting)
            else:
                flag = 0

            if debug is True:
                print ' '.join(['%.5f' % p for p in self.prob_prev])

            flag = int(flag)
            if flag > 0:
                message = "Keyword " + str(flag) + " detected at time: "
                message += time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(time.time()))
                logger.info(message)
                if True:
                    fname = os.path.join(TOP_DIR,'detections',str(uuid.uuid4().get_hex()[0:10])+'.wav')
                    wavfile.write(fname, 16000, np.int16(data_numeric))
                if True and detected_callback is not None:
                    detected_callback()
            if False and flag == 0 and self.prob_prev > 0.5:
                pdb.set_trace()
                plt.plot(data_numeric)
                #plt.show()
        logger.debug("finished.")

    def terminate(self):
        """
        Terminate audio stream. Users cannot call start() again to detect.
        :return: None
        """
        self.stream_in.stop_stream()
        self.stream_in.close()
        self.audio.terminate()

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python online_detector.py model_path [sensitivity] [audio_gain] [debug]")
        sys.exit(-1)

    # Defaults
    sensitivity = 1.5
    audio_gain = 2.0
    debug = False
    detect_rate = 0.10
    epoch = None

    model = sys.argv[1]
    if len(sys.argv) > 2:
        epoch = ast.literal_eval(sys.argv[2])
    if len(sys.argv) > 3:
        sensitivity = np.float(sys.argv[3])
    if len(sys.argv) > 4:
        audio_gain = np.float(sys.argv[4])
    if len(sys.argv) > 5:
        debug = True
    
    # capture SIGINT signal, e.g., Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    detector = Detector(model, epoch=epoch, detTh=sensitivity, audio_gain=audio_gain)
    print('Listening... Press Ctrl+C to exit')

    # main loop
    detector.start(detected_callback=play_audio_file,
                   interrupt_check=interrupt_callback,
                   sleep_time=detect_rate,debug=debug)

    detector.terminate()

