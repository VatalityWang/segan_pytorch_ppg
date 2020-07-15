import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segan.models import *
from segan.datasets import *
import soundfile as sf
from scipy.io import wavfile,loadmat,savemat
from torch.autograd import Variable
import numpy as np
import random
import librosa
import matplotlib
import timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import glob
import os
import ipdb

def read_excel_file(excelfilname):
    rate = None
    # filename='data_veu4/expanded_segan1_additive/noisy_testset/ppg_freq_HR/12-32-123.8987.xlsx'
    splitname=excelfilname.split('/')
    realname=splitname[4]
    labelhr=realname.split('-')[2][0:-5]
    labelhr=int(float(labelhr)+0.5)
    labels=np.zeros(200)
    labels[labelhr-1]=1
    ppgsignal = []
    wb = load_workbook(excelfilname)
    for sheet in wb:
        if sheet.title == "Sheet1":
            for column in sheet.columns:
                for cell in column:
            # for row in sheet.rows:
            #     for cell in row:
                    ppgsignal.append(cell.value)
    signal = np.array(ppgsignal)
    return labels, signal

def read_mat_file(filename):
    orisig=loadmat(filename)
    # for VMD
    #ppg=orisig['ppgresample']
    # for EEMD
    ppg=orisig['sig'][1]
    labels=orisig['BPM0']
    return ppg,labels
def write_mat_file(filename,ori_path,orisig):
    ORISIG=loadmat(ori_path)
    BPM0=ORISIG['BPM0']
    savemat(filename,{'cleansig':orisig,'BPM0':BPM0})
    
def write_to_excel(out_path, ppg):
    # pdb.set_trace()
    wb = Workbook()
    ws2 = wb.create_sheet("Sheet1", index=0)  # 第一个（工作簿位置）
    for i in range(0, ppg.size):
        # ws2.cell(1, i + 1, ppg[i])  # k行，i列
        ws2.cell(i+1,1,ppg[i])
    wb.save('{}'.format(out_path))  # 第j个人第k组数据
class ArgParser(object):

    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)


def main(opts):

    assert opts.cfg_file is not None
    assert opts.test_files is not None
    assert opts.g_pretrained_ckpt is not None

    with open(opts.cfg_file, 'r') as cfg_f:
        args = ArgParser(json.load(cfg_f))
        print('Loaded train config: ')
        print(json.dumps(vars(args), indent=2))
    args.cuda = opts.cuda
    if hasattr(args, 'wsegan') and args.wsegan:
        segan = WSEGAN(args)     
    else:
        segan = SEGAN(args)     
    segan.G.load_pretrained(opts.g_pretrained_ckpt, True)
    if opts.cuda:
        segan.cuda()
    segan.G.eval()
    #pdb.set_trace()
    if opts.h5:
        with h5py.File(opts.test_files[0], 'r') as f:
            twavs = f['data'][:]
    else:
        # process every wav in the test_files
        if len(opts.test_files) == 1:
            # assume we read directory
            twavs = glob.glob(os.path.join(opts.test_files[0], '*.mat'))
        else:
            # assume we have list of files in input
            twavs = opts.test_files
    print('Cleaning {} ppgs'.format(len(twavs)))
    beg_t = timeit.default_timer()

    for t_i, twav in enumerate(twavs, start=1):
        if not opts.h5:
            tbname = os.path.basename(twav)#返回最后的文件名
            # rate, wav = wavfile.read(twav)
            # pdb.set_trace()
            #labels,wav=read_excel_file(twav)
            wav,labels=read_mat_file(twav)
            # wav = normalize_wave_minmax(wav)
        else:
            tbname = 'tfile_{}.wav'.format(t_i)
            wav = twav
            twav = tbname
        #ipdb.set_trace()
        # wav = pre_emphasize(wav, args.preemph)
        pwav = torch.FloatTensor(wav).view(1,1,-1)
        labels=torch.FloatTensor(labels).view(1,-1)
        if opts.cuda:
            pwav = pwav.cuda()
        g_wav, g_c = segan.generate(pwav,labelhr=labels)
        # pdb.set_trace()   
        #tempname=tbname.split('-')
        tempname=tbname[:-4]
        newfilename=tempname+'generate.mat'
        out_path = os.path.join(opts.synthesis_path,
                                newfilename)

        #####################
        #去掉文件名后的心率值 便于循环读取
        ####################

        #生成的数据写入excel表格
        #write_to_excel(out_path,g_wav)
        write_mat_file(out_path,twav,g_wav)
        # if opts.soundfile:
        #     sf.write(out_path, g_wav, 48000)
        # else:
        #     wavfile.write(out_path, 48000, g_wav)
        end_t = timeit.default_timer()
        print('Cleaned {}/{}: {} in {} s'.format(t_i, len(twavs), twav,
                                                 end_t-beg_t))
        beg_t = timeit.default_timer()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--g_pretrained_ckpt', type=str, default=None)
    parser.add_argument('--test_files', type=str, nargs='+', default=None)
    parser.add_argument('--h5', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=111, 
                        help="Random seed (Def: 111).")
    parser.add_argument('--synthesis_path', type=str, default='segan_samples',
                        help='Path to save output samples (Def: ' \
                             'segan_samples).')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--soundfile', action='store_true', default=False)
    parser.add_argument('--cfg_file', type=str, default=None)
    parser.add_argument('--N', type=int, default=512)

    opts = parser.parse_args()

    if not os.path.exists(opts.synthesis_path):
        os.makedirs(opts.synthesis_path)
    
    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)

    main(opts)
