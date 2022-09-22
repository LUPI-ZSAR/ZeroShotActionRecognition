import numpy as np, cv2, torch
from auxiliary.auxiliary_word2vec import classes2embedding, load_word2vec

import scipy.io as io
from auxiliary.auxiliary_dataset import VideoDataset, VideoDataset_two, get_olympic_dataset, get_hmdb51_dataset, get_ucf101_dataset

def get_datasets(opt):

    if 'hmdb51' in opt.dataset:
        get_datasets = get_HMDB51_datasets(opt)

    elif 'ucf101' in opt.dataset:
        get_datasets = get_UCF101_datasets(opt)

    elif 'olympic' in opt.dataset:
        get_datasets = get_olympic_datasets(opt)

    # datasets = get_datasets(opt)
    datasets = get_datasets

    # Move datasets to dataloaders.
    dataloaders = {}
    for key, datasets in datasets.items():
        dataloader = []
        for dataset in datasets:

            dl = torch.utils.data.DataLoader(dataset,
                      batch_size= opt.bs if (opt.debug == 1) else opt.bs,
                      num_workers = 32 if (opt.debug == 1) else opt.kernels, shuffle=True, drop_last=False)
            dataloader.append(dl)
        dataloaders[key] = dataloader
    return dataloaders



def get_UCF101_datasets(opt):
    wv_model = load_word2vec()

    # TESTING ON UCF101
    train_fnames, train_labels, train_classes, test_fnames, test_labels, test_classes = get_ucf101_dataset()
    test_class_embedding, test_class_embedding1 = classes2embedding('ucf101', test_classes, wv_model)
    print('UCF101: total number of test videos {}, classes {}'.format(len(test_fnames), len(test_classes)))

    if not opt.evaluate:

        train_class_embedding, train_class_embedding1 = classes2embedding('ucf101', train_classes, wv_model)
        print('UCF101: total number of train videos {}, classes {}'.format(len(train_fnames), len(train_classes)))

        # Initialize datasets
        train_dataset = VideoDataset_two(train_fnames, train_labels, train_class_embedding, train_class_embedding1, train_classes,
                                     'kinetics%d' % len(train_classes), clip_len=opt.clip_len, n_clips=opt.n_clips,
                                     crop_size=opt.size, is_validation=False)

    n_clips = opt.n_clips if not opt.evaluate else max(5*5, opt.n_clips)
    val_dataset   = VideoDataset_two(test_fnames, test_labels, test_class_embedding, test_class_embedding1, test_classes, 'ucf101',
                                 clip_len=opt.clip_len, n_clips=n_clips, crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)

    if opt.evaluate:
        return {'training': [], 'testing': [val_dataset]}
    else:
        return {'training': [train_dataset], 'testing': [val_dataset]}


def get_HMDB51_datasets(opt):
    wv_model = load_word2vec()

    # TESTING ON HMDB51
    train_fnames, train_labels, train_classes, test_fnames, test_labels, test_classes = get_hmdb51_dataset()
    test_class_embedding, test_class_embedding1 = classes2embedding('hmdb51', test_classes, wv_model)

    print('HMDB51: total number of test videos {}, classes {}'.format(len(test_fnames), len(test_classes)))

    if not opt.evaluate:
        # TRAINING ON KINETICS
        # train_fnames, train_labels, train_classes = get_kinetics(opt.dataset)
        train_class_embedding, train_class_embedding1 = classes2embedding('hmdb51', train_classes, wv_model)
        print('HMDB51: total number of train videos {}, classes {}'.format(len(train_fnames), len(train_classes)))

        # Initialize datasets
        train_dataset = VideoDataset_two(train_fnames, train_labels, train_class_embedding,train_class_embedding1, train_classes,
                                     'kinetics%d' % len(train_classes), clip_len=opt.clip_len, n_clips=opt.n_clips,
                                     crop_size=opt.size, is_validation=False)

    n_clips = opt.n_clips if not opt.evaluate else max(5*5, opt.n_clips)

    val_dataset   = VideoDataset_two(test_fnames, test_labels, test_class_embedding, test_class_embedding1, test_classes, 'hmdb51',
                                 clip_len=opt.clip_len, n_clips=n_clips, crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)

    if opt.evaluate:
        return {'training': [], 'testing': [val_dataset]}
    else:
        return {'training': [train_dataset], 'testing': [val_dataset]}

def get_olympic_datasets(opt):
    wv_model = load_word2vec()

    # TESTING ON olympic
    train_fnames, train_labels, train_classes,test_fnames, test_labels, test_classes = get_olympic_dataset()
    test_class_embedding, test_class_embedding1 = classes2embedding('olympicsports', test_classes, wv_model)
    print(type(test_class_embedding))

    print('olympic: total number of test videos {}, classes {}'.format(len(test_fnames), len(test_classes)))

    if not opt.evaluate:

        train_class_embedding, train_class_embedding1 = classes2embedding('olympicsports', train_classes, wv_model)
        print('olympic: total number of train videos {}, classes {}'.format(len(train_fnames), len(train_classes)))

        # Initialize datasets
        train_dataset = VideoDataset_two(train_fnames, train_labels, train_class_embedding, train_class_embedding1, train_classes,
                                     'kinetics%d' % len(train_classes), clip_len=opt.clip_len, n_clips=opt.n_clips,
                                     crop_size=opt.size, is_validation=False)

    n_clips = max(8, opt.n_clips) if opt.test  else opt.n_clips # test 25 frames

    val_dataset   = VideoDataset_two(test_fnames, test_labels, test_class_embedding, test_class_embedding1, test_classes, 'olympic',
                                 clip_len=opt.clip_len, n_clips=n_clips, crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)

    if opt.evaluate:
        return {'training': [], 'testing': [val_dataset]}
    else:
        return {'training': [train_dataset], 'testing': [val_dataset]}

