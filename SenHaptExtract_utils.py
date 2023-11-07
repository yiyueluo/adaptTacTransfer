import argparse
import numpy as np
import h5py
import cv2
import random
from utils import *

def generate_states(length, num_finger, repeat_bool, space_bool):
    output = np.zeros((num_finger, length))
    list = []
    st = 0
    if space_bool: st = -1
    if repeat_bool == False:
        while len(list) < length:
            r = random.randint(st, num_finger-1)
            if r not in list: list.append(r)

    else:
        while len(list) < length:
            list.append(random.randint(st, num_finger-1))

    for i in range(len(list)):
        v = list[i]
        output[v, i] = 1

    return list, output


def tactile(ser, w, h, f, inited, viz_bool, fc):
    p = readPressure(ser, w, h)
    ts = getUnixTimestamp()
    pressure = np.asanyarray([p[0, 0], p[1, 0]])

    '''store'''
    if f != None:
        data = {'pressure': pressure, 'ts': ts, 'fc': fc}
        inited = addFrame(f, inited, data)

    '''viz'''
    if viz_bool:
        while True:
            img = viz(data, 500, 1024)
            cv2.imshow('VizualizerTouch', img)
            if cv2.waitKey(1) & 0xff == 27:
                break

    return inited, pressure


def haptic(board, pwm1_list, out1_list, pwm2_list, out2_list, act):
    amp = 1
    c = 1
    t = 5 / 1000
    if act >= 0:
        _ = vibrate(board, pwm1_list[act], out1_list[act], pwm2_list[act], out2_list[act], amp, c, t)



def record_act(ser, w, h, board, pwm1_list, out1_list, pwm2_list, out2_list, f, inited, viz_bool, fc, act):
    p = readPressure(ser, w, h)
    ts = getUnixTimestamp()
    pressure = np.asanyarray([p[0, 1], p[1, 1], p[2, 1], p[0, 0], p[1, 0], p[2, 0]])

    # ind = np.argmax(pressure)

    '''activate according to pre-defined sequence'''
    amp = 1
    c = 1
    t = 2 / 1000
    if act >= 0:
        _ = vibrate(board, pwm1_list[act], out1_list[act], pwm2_list[act], out2_list[act], amp, c, t)

    '''store'''
    if f != None:
        data = {'pressure': pressure, 'ts': ts, 'fc': fc}
        inited = addFrame(f, inited, data)

    '''viz'''
    if viz_bool:
        while True:
            img = viz(data, 500, 1024)
            cv2.imshow('VizualizerTouch', img)
            if cv2.waitKey(1) & 0xff == 27:
                break

    return inited, pressure


def extract_tactile(pressure, spacing, sec, st, space_bool):
    list = []
    list2 = []
    # for i in range(sec):
    #     m = np.mean(pressure[st+i*spacing: st+(i+1)*spacing, :], axis=0)
    #     ind = np.argmax(m)
    #     list.append(ind)
    for i in range(pressure.shape[0]):
        ind = np.argmax(pressure[i, :])
        list2.append(ind)

    for i in range(sec):
        m = np.mean(np.asanyarray(list2[st+i*spacing: st+(i+1)*spacing]))
        print (st+i*spacing, st+(i+1)*spacing, m)
        if m >= 0.5:
            list.append(1)
        else:
            list.append(0)

    if space_bool:
        output = np.zeros((pressure.shape[1]+1, len(list)))
    else:
        output = np.zeros((pressure.shape[1], len(list)))

    for i in range(len(list)):
        v = list[i]
        output[v, i] = 1

    return list, output, list2



