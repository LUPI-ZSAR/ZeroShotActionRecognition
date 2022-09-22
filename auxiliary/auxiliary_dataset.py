import os, numpy as np
from time import time

import cv2, torch
from torch.utils.data import Dataset
from auxiliary.transforms import get_transform
from scipy.spatial.distance import cdist
import random

def get_ucf101_dataset():
    dir = []
    folder = '/media/UCF101/videos/'
    for label in sorted(os.listdir(str(folder))):
            dir.append(label)
            random.shuffle(dir)
    # train_dir = dir[:51]
    # print('\n training classes:\n', train_dir)
    # test_dir = dir[-50:]
    # print('testing classes:\n', test_dir)

    train_dir = ['PlayingDhol', 'Drumming', 'BrushingTeeth', 'MoppingFloor', 'RockClimbingIndoor', 'Biking', 'Lunges', 'Diving', 'HeadMassage', 'Hammering', 'CricketBowling', 'PlayingCello', 'Nunchucks', 'HighJump', 'Knitting', 'StillRings', 'PullUps', 'FrisbeeCatch', 'RopeClimbing', 'CleanAndJerk', 'Punch', 'CliffDiving', 'TaiChi', 'CricketShot', 'PlayingSitar', 'TrampolineJumping', 'TableTennisShot', 'Kayaking', 'HandstandPushups', 'SkateBoarding', 'FieldHockeyPenalty', 'ApplyEyeMakeup', 'FloorGymnastics', 'SoccerPenalty', 'WritingOnBoard', 'JumpRope', 'HammerThrow', 'JugglingBalls', 'PizzaTossing', 'GolfSwing', 'Archery', 'ApplyLipstick', 'WallPushups', 'PlayingPiano', 'Swing', 'PlayingDaf', 'BasketballDunk', 'BlowingCandles', 'JavelinThrow', 'BalanceBeam', 'Typing']
    print('\n training classes:\n', train_dir)
    test_dir =  ['IceDancing', 'PlayingFlute', 'BaseballPitch', 'BlowDryHair', 'Rafting', 'HandstandWalking', 'Skijet', 'PlayingViolin', 'BoxingSpeedBag', 'HulaHoop', 'MilitaryParade', 'UnevenBars', 'ShavingBeard', 'CuttingInKitchen', 'BenchPress', 'PlayingTabla', 'SumoWrestling', 'ParallelBars', 'HorseRiding', 'LongJump', 'YoYo', 'ThrowDiscus', 'JumpingJack', 'BreastStroke', 'SalsaSpin', 'SkyDiving', 'BoxingPunchingBag', 'Bowling', 'HorseRace', 'Billiards', 'TennisSwing', 'VolleyballSpiking', 'BabyCrawling', 'WalkingWithDog', 'Surfing', 'Haircut', 'PommelHorse', 'Shotput', 'PoleVault', 'SoccerJuggling', 'Basketball', 'Rowing', 'Skiing', 'FrontCrawl', 'Mixing', 'BodyWeightSquats', 'Fencing', 'PlayingGuitar', 'BandMarching', 'PushUps']
    print('testing classes:\n', test_dir)

    fnames_train, labels_train = [], []
    for label in train_dir:
        for fname in os.listdir(os.path.join(str(folder), label)):
            fnames_train.append(os.path.join(str(folder), label, fname))
            labels_train.append(label)

    classes_train = np.unique(labels_train)

    fnames_test, labels_test = [], []
    for label in test_dir:
        for fname in os.listdir(os.path.join(str(folder), label)):
            fnames_test.append(os.path.join(str(folder), label, fname))
            labels_test.append(label)

    classes_test = np.unique(labels_test)
    return fnames_train, labels_train, classes_train, fnames_test,labels_test,classes_test

def get_hmdb51_dataset():
    dir=[]
    folder = '/media/HMDB51/videos/'
    for label in sorted(os.listdir(str(folder))):
        dir.append(label)
        random.shuffle(dir)
    # train_dir = dir[:26]
    # print('\n training classes:\n', train_dir)
    # test_dir = dir[-25:]
    # print('testing classes:\n', test_dir)

    train_dir = ['jump', 'sword_exercise', 'golf', 'brush_hair', 'ride_horse', 'run', 'smile', 'climb', 'laugh', 'talk', 'somersault', 'chew', 'pour', 'climb_stairs', 'wave', 'pick', 'shoot_gun', 'ride_bike', 'throw', 'smoke', 'handstand', 'clap', 'hug', 'sit', 'swing_baseball', 'draw_sword']
    print('\n training classes:\n', train_dir)
    test_dir = ['shake_hands', 'catch', 'turn', 'punch', 'eat', 'shoot_ball', 'hit', 'kiss', 'shoot_bow', 'dive', 'walk', 'sword', 'fencing', 'kick_ball', 'cartwheel', 'pushup', 'dribble', 'kick', 'pullup', 'flic_flac', 'push', 'stand', 'fall_floor', 'drink', 'situp']
    print('testing classes:\n', test_dir)

    fnames_train, labels_train = [], []
    for label in train_dir:
        dir = os.path.join(str(folder), label)
        if not os.path.isdir(dir): continue
        for fname in sorted(os.listdir(dir)):
            if fname[-4:] != '.avi':
                continue
            fnames_train.append(os.path.join(str(folder), label, fname))
            labels_train.append(label.replace('_', ' '))

    fnames_train, labels_train = np.array(fnames_train), np.array(labels_train)
    classes_train = np.unique(labels_train)

    fnames_test, labels_test = [], []
    for label in test_dir:
        dir = os.path.join(str(folder), label)
        if not os.path.isdir(dir): continue
        for fname in sorted(os.listdir(dir)):
            if fname[-4:] != '.avi':
                continue
            fnames_test.append(os.path.join(str(folder), label, fname))
            labels_test.append(label.replace('_', ' '))

    fnames_test, labels_test = np.array(fnames_test), np.array(labels_test)
    classes_test = np.unique(labels_test)
    return fnames_train, labels_train, classes_train, fnames_test,labels_test,classes_test


def get_olympic_dataset():
    dir = []
    folder = '/media/OlympicSports/videos/'
    for label in sorted(os.listdir(str(folder))):
        dir.append(label)
        random.shuffle(dir)
    # train_dir = dir[:8]
    # print('\n training classes:\n', train_dir)
    # test_dir = dir[-8:]
    # print('testing classes:\n', test_dir)

    train_dir =['tennis_serve', 'triple_jump', 'discus_throw', 'shot_put', 'basketball_layup', 'long_jump', 'high_jump', 'javelin_throw']

    print('\n training classes:\n', train_dir)
    test_dir = ['clean_and_jerk', 'hammer_throw', 'pole_vault', 'diving_springboard_3m', 'diving_platform_10m', 'vault', 'snatch', 'bowling']


    print('testing classes:\n', test_dir)

    fnames_train, labels_train = [], []
    for label in train_dir:
        dir = os.path.join(str(folder), label)
        if not os.path.isdir(dir): continue
        for fname in sorted(os.listdir(dir)):
            if fname[-4:] != '.avi':
                continue
            fnames_train.append(os.path.join(str(folder), label, fname))
            labels_train.append(label.replace('_', ' '))

    fnames_train, labels_train = np.array(fnames_train), np.array(labels_train)
    classes_train = np.unique(labels_train)

    fnames_test, labels_test = [], []
    for label in test_dir:
        dir = os.path.join(str(folder), label)
        if not os.path.isdir(dir): continue
        for fname in sorted(os.listdir(dir)):
            if fname[-4:] != '.avi':
                continue
            fnames_test.append(os.path.join(str(folder), label, fname))
            labels_test.append(label.replace('_', ' '))

    fnames_test, labels_test = np.array(fnames_test), np.array(labels_test)
    classes_test = np.unique(labels_test)
    return fnames_train, labels_train, classes_train, fnames_test,labels_test,classes_test

"""========================================================="""


def load_clips_tsn(fname, clip_len=16, n_clips=1, is_validation=False):
    if not os.path.exists(fname):
        print('Missing: '+fname)
        return []
    # initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count == 0 or frame_width == 0 or frame_height == 0:
        print('loading error, switching video ...')
        print(fname)
        return []

    total_frames = frame_count #min(frame_count, 300)
    sampling_period = max(total_frames // n_clips, 1)
    n_snipets = min(n_clips, total_frames // sampling_period)
    if not is_validation:
        starts = np.random.randint(0, max(1, sampling_period - clip_len), n_snipets)
    else:
        starts = np.zeros(n_snipets)
    offsets = np.arange(0, total_frames, sampling_period)
    selection = np.concatenate([np.arange(of+s, of+s+clip_len) for of, s in zip(offsets, starts)])

    frames = []
    count = ret_count = 0
    while count < selection[-1]+clip_len:
        retained, frame = capture.read()
        if count not in selection:
            count += 1
            continue
        if not retained:
            if len(frames) > 0:
                frame = np.copy(frames[-1])
            else:
                frame = (255*np.random.rand(frame_height, frame_width, 3)).astype('uint8')
            frames.append(frame)
            ret_count += 1
            count += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    capture.release()
    frames = np.stack(frames)
    total = n_clips * clip_len
    while frames.shape[0] < total:
        frames = np.concatenate([frames, frames[:(total - frames.shape[0])]])
    frames = frames.reshape([n_clips, clip_len, frame_height, frame_width, 3])
    return frames


class VideoDataset(Dataset):

    def __init__(self, fnames, labels, class_embed, classes, name, load_clips=load_clips_tsn,
                 clip_len=8, n_clips=1, crop_size=112, is_validation=False, evaluation_only=False):
        if 'kinetics' in name:
            fnames, labels = self.clean_data(fnames, labels)
        self.data = fnames
        self.labels = labels
        self.class_embed = class_embed
        self.class_name = classes
        self.name = name

        self.clip_len = clip_len
        self.n_clips = n_clips

        self.crop_size = crop_size  # 112
        self.is_validation = is_validation

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        self.transform = get_transform(self.is_validation, crop_size)
        self.loadvideo = load_clips

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label_array[idx]
        buffer = self.loadvideo(sample, self.clip_len, self.n_clips, self.is_validation)
        if len(buffer) == 0:
            buffer = np.random.rand(self.n_clips, 3, self.clip_len, 112, 112).astype('float32')
            buffer = torch.from_numpy(buffer)
            return buffer, -1, self.class_embed[0], -1
        s = buffer.shape
        buffer = buffer.reshape(s[0] * s[1], s[2], s[3], s[4])
        buffer = torch.stack([torch.from_numpy(im) for im in buffer], 0)
        buffer = self.transform(buffer)
        buffer = buffer.reshape(3, s[0], s[1], self.crop_size, self.crop_size).transpose(0, 1)
        return buffer, label, self.class_embed[label], idx

    def __len__(self):
        return len(self.data)

    @staticmethod
    def clean_data(fnames, labels):
        if not isinstance(fnames[0], str):
            print('Cannot check for broken videos')
            return fnames, labels
        broken_videos_file = 'assets/kinetics_broken_videos.txt'
        if not os.path.exists(broken_videos_file):
            print('Broken video list does not exists')
            return fnames, labels

        t = time()
        with open(broken_videos_file, 'r') as f:
            broken_samples = [r[:-1] for r in f.readlines()]
        data = [x[75:] for x in fnames]
        keep_sample = np.in1d(data, broken_samples) == False
        fnames = np.array(fnames)[keep_sample]
        labels = np.array(labels)[keep_sample]
        print('Broken videos %.2f%% - removing took %.2f' % (100 * (1.0 - keep_sample.mean()), time() - t))
        return fnames, labels




class VideoDataset_two(Dataset):
    
    def __init__(self, fnames, labels, class_embed, class_embed1, classes, name, load_clips=load_clips_tsn,
                 clip_len=8, n_clips=1, crop_size=112, is_validation=False, evaluation_only=False):
        if 'kinetics' in name:
            fnames, labels = self.clean_data(fnames, labels)
        self.data = fnames
        self.labels = labels
        self.class_embed = class_embed
        self.class_embed1 = class_embed1
        self.class_name = classes
        self.name = name

        self.clip_len = clip_len
        self.n_clips = n_clips

        self.crop_size = crop_size  # 112
        self.is_validation = is_validation

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        self.transform = get_transform(self.is_validation, crop_size)
        self.loadvideo = load_clips

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label_array[idx]
        buffer = self.loadvideo(sample, self.clip_len, self.n_clips, self.is_validation)
        if len(buffer) == 0:
            buffer = np.random.rand(self.n_clips, 3, self.clip_len, 112, 112).astype('float32')
            buffer = torch.from_numpy(buffer)
            return buffer, -1, self.class_embed[0], self.class_embed1[0] -1
        s = buffer.shape
        buffer = buffer.reshape(s[0] * s[1], s[2], s[3], s[4])
        buffer = torch.stack([torch.from_numpy(im) for im in buffer], 0)
        buffer = self.transform(buffer)
        buffer = buffer.reshape(3, s[0], s[1], self.crop_size, self.crop_size).transpose(0, 1)
        return buffer, label, self.class_embed[label], self.class_embed1[label],idx

    def __len__(self):
        return len(self.data)

    @staticmethod
    def clean_data(fnames, labels):
        if not isinstance(fnames[0], str):
            print('Cannot check for broken videos')
            return fnames, labels
        broken_videos_file = 'assets/kinetics_broken_videos.txt'
        if not os.path.exists(broken_videos_file):
            print('Broken video list does not exists')
            return fnames, labels

        t = time()
        with open(broken_videos_file, 'r') as f:
            broken_samples = [r[:-1] for r in f.readlines()]
        data = [x[75:] for x in fnames]
        keep_sample = np.in1d(data, broken_samples) == False
        fnames = np.array(fnames)[keep_sample]
        labels = np.array(labels)[keep_sample]
        print('Broken videos %.2f%% - removing took %.2f' % (100 * (1.0 - keep_sample.mean()), time() - t))
        return fnames, labels

