import numpy as np
import h5py
import serial
from pyfirmata import ArduinoMega
from pyfirmata import INPUT, OUTPUT, PWM
import sys, os, re, time, shutil, math, random, datetime, argparse, signal
import multiprocessing as mp
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def tactile_reading(path=''):
    f = h5py.File(path, 'r')
    fc = np.int16(np.amax(f['fc'])) + 1
    ts = np.array(f['ts'][:fc])
    pressure = np.array(f['pressure'][:fc]).astype(np.float32)
    return pressure, ts, fc

def initSensor(sen_port):
    ser = serial.Serial(sen_port, baudrate=500000, timeout=1.0)
    assert ser.is_open, 'Failed to open sensing COM port!'

    return ser

def initThread(sen_port, hapt_port):
    # Setup input
    ser = serial.Serial(sen_port, baudrate=500000, timeout=1.0)
    assert ser.is_open, 'Failed to open sensing COM port!'

    '''set haptic'''
    P_A11 = 2
    P_A12 = 22
    P_B11 = 4
    P_B12 = 26
    P_C11 = 6
    P_C12 = 30
    P_D11 = 8
    P_D12 = 34

    P_A21 = 3
    P_A22 = 24
    P_B21 = 5
    P_B22 = 28
    P_C21 = 7
    P_C22 = 32
    P_D21 = 9
    P_D22 = 36

    port = hapt_port
    board = ArduinoMega(port)

    board.digital[P_A11].mode = PWM
    board.digital[P_A21].mode = PWM
    board.digital[P_B11].mode = PWM
    board.digital[P_B21].mode = PWM
    board.digital[P_C11].mode = PWM
    board.digital[P_C21].mode = PWM
    board.digital[P_D11].mode = PWM
    board.digital[P_D21].mode = PWM

    board.digital[P_A12].mode = OUTPUT
    board.digital[P_A22].mode = OUTPUT
    board.digital[P_B12].mode = OUTPUT
    board.digital[P_B22].mode = OUTPUT
    board.digital[P_C12].mode = OUTPUT
    board.digital[P_C22].mode = OUTPUT
    board.digital[P_D12].mode = OUTPUT
    board.digital[P_D22].mode = OUTPUT

    ''' user study/tactile occlusion/teleoperation parameter'''
    # pwm1_list = [P_A11, P_A11, P_A11, P_D11, P_D11, P_D11]
    # out1_list = [P_A22, P_B22, P_D22, P_A22, P_C22, P_D22]
    # pwm2_list = [P_A21, P_B21, P_D21, P_A21, P_C21, P_D21]
    # out2_list = [P_A12, P_A12, P_A12, P_D12, P_D12, P_D12]

    ''' teleoperation glove parameter'''
    # pwm1_list = [P_A11, P_A11, P_A11, P_D11, P_D11, P_D11]
    # out1_list = [P_A22, P_B22, P_D22, P_A22, P_C22, P_D22]
    # pwm2_list = [P_A21, P_B21, P_D21, P_A21, P_C21, P_D21]
    # out2_list = [P_A12, P_A12, P_A12, P_D12, P_D12, P_D12]

    # pwm1_list = [P_A11, P_A11, P_A11, P_D11, P_D11, P_D11]
    # out1_list = [P_A22, P_B22, P_D22, P_C22, P_A22, P_D22]
    # pwm2_list = [P_A21, P_B21, P_D21, P_C21, P_A21, P_D21]
    # out2_list = [P_A12, P_A12, P_A12, P_D12, P_D12, P_D12]

    '''test parameter'''
    # pwm1_list = [P_D11, P_A11]
    # out1_list = [P_D22, P_D22]
    # pwm2_list = [P_D21, P_D21]
    # out2_list = [P_D12, P_A12]

    '''piano parameter'''
    pwm1_list = [P_A11, P_A11, P_A11, P_A11, P_D11]
    out1_list = [P_A22, P_B22, P_C22, P_D22, P_D22]
    pwm2_list = [P_A21, P_B21, P_C21, P_D21, P_D21]
    out2_list = [P_A12, P_A12, P_A12, P_A12, P_D12]


    return ser, board, pwm1_list, out1_list, pwm2_list, out2_list

def readPressure(ser, w, h):
    # Request readout
    ser.reset_input_buffer() # Remove the confirmation 'w' sent by the sensor
    ser.write('a'.encode('utf-8')) # Request data from the sensor

    # Receive data
    length = 2 * w * h
    input_string = ser.read(length)
    x = np.frombuffer(input_string, dtype=np.uint8).astype(np.uint16)
    if not len(input_string) == length:
        # self.log("Only got %d values => Drop frame." % len(input_string))
        return None

    x = x[0::2] * 32 + x[1::2]
    x = x.reshape(h, w).transpose(1, 0)
    # print (x)
    return x

def vibrate(board, pwm1, out1, pwm2, out2, amp, c, t):
    for i in range(c):
        board.digital[pwm1].write(amp)
        board.digital[out1].write(1)
        time.sleep(t)
        board.digital[pwm1].write(0)
        board.digital[out1].write(0)

        board.digital[pwm2].write(amp)
        board.digital[out2].write(1)
        time.sleep(t)
        board.digital[pwm2].write(0)
        board.digital[out2].write(0)

    return None


def getUnixTimestamp():
    return np.datetime64(datetime.datetime.now()).astype(np.int64) / 1e6  # unix TS in secs and microsecs

def fitImageToBounds(img, bounds, upscale = False, interpolation = cv2.INTER_LINEAR):
    inAsp = img.shape[1] / img.shape[0]
    outAsp = bounds[0] / bounds[1]

    if not upscale and img.shape[1] <= bounds[0] and img.shape[0] <= bounds[1]:
        return img
    elif img.shape[1] == bounds[0] and img.shape[0] == bounds[1]:
        return img

    if inAsp < outAsp:
        # Narrow to wide
        height = bounds[1]
        width = math.floor(inAsp * height+ 0.5)
    else:
        width = bounds[0]
        height = math.floor(width / inAsp + 0.5)

    res = cv2.resize(img, (int(width), int(height)), interpolation = interpolation)
    if len(res.shape) < len(img.shape):
        res = res[..., np.newaxis]
    return res

def letterBoxImage(img, size, return_bbox = False):
    # letter box
    szIn = np.array([img.shape[1], img.shape[0]])
    x0 = (size - szIn) // 2
    x1 = x0 + szIn

    res = np.zeros([size[1], size[0], img.shape[2]], img.dtype)
    res[x0[1]:x1[1],x0[0]:x1[0],:] = img

    if return_bbox:
        return res, np.concatenate((x0,x1-x0))

    return res

def resizeImageLetterBox(img, size, interpolation = cv2.INTER_LINEAR, return_bbox = False):
    img = fitImageToBounds(img, size, upscale = True, interpolation = interpolation)
    return letterBoxImage(img, size, return_bbox)

def viz(data, min, max):
    p, ts, fc = data['pressure'], data['ts'], data['fc']
    resolution = (600, 600)
    pressure = (p.astype(np.float32) - min) / (max - min)
    pressure = np.clip(pressure, 0, 1)
    im = cv2.applyColorMap((np.clip(pressure, 0, 1) * 255).astype('uint8'), cv2.COLORMAP_JET)
    im = fitImageToBounds(im, resolution, upscale=True, interpolation=cv2.INTER_NEAREST)
    caption = '[%s] %06d (%.3f s)|Range=%03d(%03d)-%03d(%03d)' % (
        'touch', fc, ts, p.min(), min, p.max(), max)
    cv2.putText(im, caption, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    im = resizeImageLetterBox(im, resolution, interpolation=cv2.INTER_NEAREST)
    return im


def iniateStorage(outputPath, name):
    hdf5Filename = os.path.join(outputPath, '%s.hdf5' % name)
    print('Output file: %s' % hdf5Filename)
    f = h5py.File(hdf5Filename, 'w')
    inited = False

    return f, inited

def addFrame(f, inited, data):
    blockSize = 1024
    frameCount = data['fc']
    if not inited:
        # Initialize datasets
        # f.create_dataset('frame_count', (1,), dtype=np.uint32)
        # f['frame_count'][0] = 0
        # # self.f.create_dataset('ts', (self.blockSize,), maxshape = (None,), dtype = np.float64)

        for k, v in data.items():
            if np.isscalar(v):
                v = np.array([v])
                sz = [blockSize, ]
            else:
                v = np.array(v)
                sz = [blockSize, *v.shape]
            maxShape = sz.copy()
            maxShape[0] = None
            f.create_dataset(k, tuple(sz), maxshape=tuple(maxShape), dtype=v.dtype)

        inited = True

    # Check size
    oldSize = f['ts'].shape[0]
    if oldSize == frameCount:
        newSize = oldSize + blockSize

        # self.f['ts'].resize(newSize, axis=0)
        for k, v in data.items():
            f[k].resize(newSize, axis=0)

    # Append data
    # self.f['ts'][self.frameCount] = ts
    # for k, v in data.items():
    #     f[k].append = v
    #     print ('here')

    for k, v in data.items():
        f[k][frameCount, ...] = v

    # # Note frame count
    # frameCount += 1
    # f['frame_count'][0] = self.frameCount

    # Flush to prevent data loss
    f.flush()

    # print (f['pressure'][0].shape)

    return inited


