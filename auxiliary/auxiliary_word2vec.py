import os, numpy as np
import torch
from time import time
from gensim.models import KeyedVectors as Word2Vec
from sklearn.preprocessing import normalize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
# import nltk
# nltk.download('wordnet')


def classes2embedding(dataset_name, class_name_inputs, wv_model):
    if dataset_name == 'ucf101':
        one_class2embed = one_class2embed_ucf_label
        one_class2embed1 = one_class2embed_ucf_object
    elif dataset_name == 'hmdb51':
        one_class2embed = one_class2embed_hmdb_label
        one_class2embed1 = one_class2embed_hmdb_object
    elif dataset_name == 'olympicsports':
        one_class2embed = one_class2embed_olympic_label
        one_class2embed1 = one_class2embed_olympic_object
    embedding = [one_class2embed(class_name,wv_model)[0] for class_name in class_name_inputs]
    embedding = np.stack(embedding)

    embedding1 = [one_class2embed1(class_name, wv_model)[0] for class_name in class_name_inputs]
    embedding1 = np.stack(embedding1)

    return normalize(embedding.squeeze()), normalize(embedding1.squeeze())


def load_word2vec():
    try:
        wv_model = Word2Vec.load('./assets/GoogleNews', mmap='r')
    except:
        wv_model = Word2Vec.load_word2vec_format(
                '/home/GoogleNews-vectors-negative300.bin', binary=True)
        wv_model.init_sims(replace=True)
        wv_model.save('./assets/GoogleNews')
    return wv_model


def one_class2embed_ucf(name, wv_model):
    change = {
        'CleanAndJerk': ['weight', 'lift'],
        'Skijet': ['Skyjet'],
        'HandStandPushups': ['handstand', 'pushups'],
        'HandstandPushups': ['handstand', 'pushups'],
        'PushUps': ['pushups'],
        'PullUps': ['pullups'],
        'WalkingWithDog': ['walk', 'dog'],
        'ThrowDiscus': ['throw', 'disc'],
        'TaiChi': ['taichi'],
        'CuttingInKitchen': ['cut', 'kitchen'],
        'YoYo': ['yoyo'],
    }
    if name in change:
        name_vec = change[name]
    else:
        upper_idx = np.where([x.isupper() for x in name])[0].tolist()
        upper_idx += [len(name)]
        name_vec = []
        for i in range(len(upper_idx)-1):
            name_vec.append(name[upper_idx[i]: upper_idx[i+1]])
        name_vec = [n.lower() for n in name_vec]
        name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec


def one_class2embed_ucf_label(name, wv_model):
    change = {
        'CleanAndJerk': ['weight', 'lift'],
        'Skijet': ['Skyjet'],
        'HandStandPushups': ['handstand', 'pushups'],
        'HandstandPushups': ['handstand', 'pushups'],
        'PushUps': ['pushups'],
        'PullUps': ['pullups'],
        'WalkingWithDog': ['walk', 'dog'],
        'ThrowDiscus': ['throw', 'disc'],
        'TaiChi': ['taichi'],
        'CuttingInKitchen': ['cut', 'kitchen'],
        'YoYo': ['yoyo'],
    }
    if name in change:
        name_vec = change[name]
    else:
        upper_idx = np.where([x.isupper() for x in name])[0].tolist()
        upper_idx += [len(name)]
        name_vec = []
        for i in range(len(upper_idx)-1):
            name_vec.append(name[upper_idx[i]: upper_idx[i+1]])
        name_vec = [n.lower() for n in name_vec]
        name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec


def one_class2embed_ucf_object(name, wv_model):
    change = {
        'ApplyEyeMakeup' : ['beauty', 'paint','war','makeup', 'make'],
        'ApplyLipstick': ['lip', 'girl', 'smoking', 'woman', 'call'],
        'Archery': ['bow', 'golf', 'arrow', 'archery', 'archer'],
        'BabyCrawling' : ['potty', 'floor', 'great', 'room', 'day'],
        'BalanceBeam' : ['bars', 'horse', 'beam', 'uneven', 'parallel'],
        'BandMarching' : ['guard', 'majorette', 'band', 'honor', 'military'],
        'BaseballPitch' : ['hitter', 'baseball', 'base', 'game', 'home'],
        'Basketball' : ['basketball','court', 'game', 'tennis', 'backboard'],
        'BasketballDunk': ['basketball', 'game', 'court', 'backboard', 'volleyball'],
        'BenchPress' : ['weight', 'bench', 'spa', 'health', 'gym'],
        'Biking' :['bike', 'bicycle', 'safety', 'wheel', 'cycle'],
        'Billiards' :['billiard','table','ball','room','pool'],
        'BlowDryHair' :['hair','beauty','salon','blow','drier'],
        'BlowingCandles' : ['light','cake','candle','tester','taste'],
        'BodyWeightSquats':['weight','treadmill','exercise','gym','gymnastic'],
        'Bowling' : ['bowling','ball','pin','skittle','equipment'],
        'BoxingPunchingBag':['bag','punching','weight','boxing','sparring'],
        'BoxingSpeedBag':['bag','punching','weight','room','ball'],
        'BreastStroke' :['skin','water','diving','snorkel','sea'],
        'BrushingTeeth': ['toothbrush','tester','taste','oral','gum'],
        'CleanAndJerk' : ['weight','gymnastic','bench','gym','exercise'],
        'CliffDiving': ['cliff', 'formation', 'diving', 'coast', 'water', 'drop'],
        'CricketBowling' : ['game', 'cricket','court', 'squash','tennis'],
        'CricketShot':['game','tennis','cricket','golf','court'],
        'CuttingInKitchen' : ['knife', 'julienne','cheese','bread','dough'],
        'Diving' : ['water','resort','table','deck','spa'],
        'Drumming' : ['drum','instrument', 'tom','snare','television'],
        'Fencing' : ['horse','fencing','bars','sword','foil'],
        'FieldHockeyPenalty' : ['tennis','game','football','court','field'],
        'FloorGymnastics':['tennis','bars','horse','table','court'],
        'FrisbeeCatch': ['football','game','sport','rugby','field'],
        'FrontCrawl': ['water','diving','skin','snorkel','dive'],
        'GolfSwing':['golf','club','iron','play','head'],
        'Haircut' : ['beauty','hair','salon','stylist','styler'],
        'Hammering':['saw','joint','carpenter','mortise','table'],
        'HammerThrow':['tennis','game','basketball','court','jump'],
        'HandstandPushups':['weight','bar','bars','tumbling','acrobatics'],
        'HandstandWalking' : ['game','weight','room','television','jump'],
        'HeadMassage' : ['beauty','coiffeur', 'barbershop','person','stylist'],
        'HighJump' : ['tennis','court','jump','sport','field'],
        'HorseRace' : ['racing','horse','football','starting','stadium'],
        'HorseRiding' : ['horse','riding','walking','pony','hack'],
        'HulaHoop' : ['hoop','dancer','twirler','ballet','tightrope'],
        'IceDancing' : ['ice','skating','skate','rink','speed'],
        'JavelinThrow' : ['tennis','game','jump','sport','field'],
        'JugglingBalls' : ['room','table','man','human','homo'],
        'JumpingJack' : ['weight','room','ball','gallery','suit'],
        'JumpRope' : ['squash','weight','game','gymnastic','court'],
        'Kayaking' : ['water','paddle','boat','kayak','sport'],
        'Knitting' : ['needle','knot','purl','crochet','stitch'],
        'LongJump' : ['jump','game','sport','field','track'],
        'Lunges' : ['weight','gym','tennis','bench','treadmill'],
        'MilitaryParade' : ['military','guard','dress','honor','recruit'],
        'Mixing' : ['sauce','cream','batter','bechamel','pan'],
        'MoppingFloor' : ['cleaning','mop','room','woman','cleaner'],
        'Nunchucks' : ['table','golf','room','sparring','homo'],
        'ParallelBars' : ['bars','horse','uneven','bar','parallel'],
        'PizzaTossing' : ['table','room','shop','machine','human'],
        'PlayingCello' : ['bass','instrument','viol','fiddle','viola'],
        'PlayingDaf' : ['instrument','drum','tam','gong','percussionist'],
        'PlayingDhol' : ['drum','instrument','bongo','cart','percussive'],
        'PlayingFlute' : ['flute','instrument','woodwind','wind','reed'],
        'PlayingGuitar' : ['guitar','instrument','music','resonator','transcriber'],
        'PlayingPiano' : ['piano','grand','concert','action','clavier'],
        'PlayingSitar' : ['instrument','guitar','sitar','resonator','cittern'],
        'PlayingTabla' : ['instrument','sitar','drum','player','bongo'],
        'PlayingViolin' : ['instrument','violin','stringed','viola','fiddle'],
        'PoleVault' : ['tennis','pole','jump','sport','field'],
        'PommelHorse' : ['horse','bars','beam','gymnastic','uneven'],
        'PullUps' : ['bar','weight','bars','pole','parallel'],
        'Punch' : ['lightweight','sparring','heavyweight','boxing','flyweight'],
        'PushUps' : ['weight','wrestling','gymnastic','bench','exercise'],
        'Rafting' : ['boat','water','paddle','sport','raft'],
        'RockClimbingIndoor' : ['rock','climbing','climber','gusset','abseiler'],
        'RopeClimbing' : ['pole','bars','bar','vaulter','uneven'],
        'Rowing' : ['boat','racing','rowing','sweep','shell'],
        'SalsaSpin' : ['dance','hall','dancer','skating','master'],
        'ShavingBeard' : ['man','smoker','shaving','tester','taste'],
        'Shotput' : ['game','jump','tennis','court','sport'],
        'SkateBoarding' : ['skate','skating','speed','rollerblading','skateboarding'],
        'Skiing' : ['ski','skiing','slope','ice','piste'],
        'Skijet' : ['water','boat','whale','scooter','sea'],
        'SkyDiving' : ['drogue','parachute','chute','ripcord','diving'],
        'SoccerJuggling' : ['game','tennis','court','basketball','ball'],
        'SoccerPenalty' : ['football','game','tennis','stadium','court'],
        'StillRings' : ['bars','horse','pole','bar','uneven'],
        'SumoWrestling' : ['sumo','wrestling','heavyweight','ring','rassling'],
        'Surfing' : ['wave','coast','water','bomb','littoral'],
        'Swing' : ['oak','playground','swing','tree','bar'],
        'TableTennisShot' : ['tennis','table','room','billiard','pingpong'],
        'TaiChi' : ['golf','club','game','play','sparring'],
        'TennisSwing' : ['tennis','court','squash','player','professional'],
        'ThrowDiscus' : ['tennis','game','hitter','cage','court'],
        'TrampolineJumping' : ['tennis','trampoline','jump','golf','table'],
        'Typing' : ['computer','keyboard','key','device','system'],
        'UnevenBars' : ['bars','pole','horse','uneven','parallel'],
        'VolleyballSpiking' : ['basketball','game','volleyball','squash','court'],
        'WalkingWithDog' : ['dog','terrier','golf','canis','domestic'],
        'WallPushups' : ['weight','treadmill','punching','gym','bag'],
        'WritingOnBoard' : ['teacher','instructor','decorator','human','reader'],
        'YoYo' : ['tennis','human','homo','man','table']
    }
    if name in change:
        name_vec = change[name]
    else:
        upper_idx = np.where([x.isupper() for x in name])[0].tolist()
        upper_idx += [len(name)]
        name_vec = []
        for i in range(len(upper_idx)-1):
            name_vec.append(name[upper_idx[i]: upper_idx[i+1]])
        name_vec = [n.lower() for n in name_vec]
        name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec


#####################################################hmdb51################################################
def one_class2embed_hmdb(name, wv_model):
    change = {'claping': ['clapping']}
    if name in change:
        name_vec = change[name]
    else:
        name_vec = name.split(' ')
    name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec

def one_class2embed_hmdb_label(name, wv_model):
    change = {'claping': ['clapping']}
    if name in change:
        name_vec = change[name]
    else:
        name_vec = name.split(' ')
    name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec


def one_class2embed_hmdb_object(name, wv_model):
    change = {
        'brush hair' : ['hair', 'room','beauty','dressing', 'woman'],
        'cartwheel': ['game', 'skating', 'squash', 'jump', 'bars'],
        'catch': ['game', 'footbal', 'sport', 'golf', 'outdoor'],
        'chew' : ['person', 'woman', 'girl', 'hair', 'young'],
        'clap' : ['television', 'tv', 'room', 'man', 'speaker'],
        'climb' : ['rock', 'climber', 'climbing', 'bungee', 'cliff'],
        'climb stairs' : ['stair', 'tread', 'step', 'rollerblading', 'skateboarding'],
        'dive' : ['television','bungee', 'diving', 'cliff', 'water'],
        'draw sword': ['sword', 'hall', 'gallery', 'sapiens', 'blade'],
        'dribble' : ['basketball', 'game', 'court', 'squash', 'hoop'],
        'drink' :['drinker', 'man', 'person', 'smoking', 'human'],
        'eat' :['man','human','person','smoking','tester'],
        'fall floor' :['man','television','human','sapiens','homo'],
        'fencing' : ['sword','fencing','sparring','television','saber'],
        'flic flac':['game','jump','bars','golf','squash'],
        'golf' : ['golf','club','iron','play','head'],
        'hand stand':['game','golf','tennis','court','beam'],
        'hug':['television','skating','crossing','human','reporter'],
        'hit' :['car','vehicle','golf','drum','room'],
        'jump': ['tennis','game','football','television','crossbar'],
        'kick' : ['sapiens','homo','sparring','human','man'],
        'kick ball': ['football','ball','game','television','tube'],
        'kiss' : ['partner', 'kisser','mate', 'sweetie','steady'],
        'laugh':['man','human','black','sister','person'],
        'pick' : ['golf', 'garbage','room','house','crossing'],
        'pour' : ['drinker','tester','taste','bar','beer'],
        'pullup' : ['weight','bar', 'pole','gym','high'],
        'punch' : ['punching','sparring','bag','boxing','weight'],
        'push' : ['cart','car','room','potty','table'],
        'pushup':['weight','bench','therapist','man','gymnastic'],
        'ride bike': ['bike','bicycle','safety','wheel','cycle'],
        'ride horse': ['horse','riding','saddle','pony','horseback'],
        'run':['man','human','football','game','homo'],
        'shake hand' : ['television','interlocutor','speaker','reporter','conversational'],
        'shoot ball':['basketball','court','game','backboard','tennis'],
        'shoot bow':['bow','arrow','archer','golf', 'archery'],
        'shoot gun':['gun','automatic','rifle','arm','firearm'],
        'sit' : ['room','table','man','television','homo'],
        'sit up' : ['weight','television', 'room','therapist','wrestling'],
        'smile' : ['person','man','woman','black','girl'],
        'smoke' : ['man','smoking','human','smoker','woman'],
        'somersault' : ['game','court','bars','basketball','volleyball'],
        'stand' : ['room','man','table','television','human'],
        'swing baseball' : ['baseball','hitter','game','tennis','base'],
        'sword' : ['television','sword','hall','game','sapiens'],
        'sword exercise' : ['hall','sword','stick','fencing','squash'],
        'talk' : ['man','sapiens','human','homo','television'],
        'throw' : ['hitter','game','baseball','base','first'],
        'turn' : ['man','human','person','television','witness'],
        'walk' : ['room','television','window','man','human'],
        'wave' : ['person','suit','room','black','television']
                
    }
    if name in change:
        name_vec = change[name]
    else:
        name_vec = name.split(' ')
    name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec


def one_class2embed_olympic_label(name, wv_model):
    change = {
        'clean and jerk': ['weight', 'lift'],
        # 'discus throw': [],
        'diving platform 10m':['diving', 'platform'],
        'diving springboard 3m': ['diving', 'springboard']
    }
    if name in change:
        name_vec = change[name]
    else:
        name_vec = name.split(' ')
    name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec


def one_class2embed_olympic_object(name, wv_model):
    change = {
       'basketball layup':['basketball', 'court', 'game', 'backboard', 'squash'],
        'bowling':['bowling', 'ball', 'pin', 'skittle', 'hall'],
        'clean and jerk': ['weight', 'gym', 'bench', 'gymnastic', 'exercise'],
        'discus throw': ['cage', 'tennis', 'game', 'volleyball', 'hitter'],
        'diving platform 10m':['bars', 'hall', 'beam', 'solar', 'mezzanine'],
        'diving springboard 3m':['table', 'water', 'bars', 'volleyball', 'game'],
        'hammer throw':['court', 'cage', 'tennis', 'game', 'basketball'],
        'high jump':['jump', 'sport', 'tennis', 'field', 'game'],
        'javelin throw':['tennis', 'game', 'sport', 'jump', 'field'],
        'long jump':['sport', 'track', 'jump', 'tennis', 'field'],
        'pole vault':['pole', 'jump', 'vaulter', 'tennis', 'game'],
        'shot put':['game', 'jump', 'sport', 'field', 'track'],
        'snatch':['weight', 'table', 'gymnastic', 'gym', 'bench'],
        'tennis serve':['tennis', 'court', 'squash', 'racquet', 'player'],
        'triple jump':['jump', 'track', 'field', 'sport', 'game'],
        'vault':['horse', 'beam', 'gymnastic', 'bars', 'jump']
    }
    if name in change:
        name_vec = change[name]
    else:
        name_vec = name.split(' ')
    name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec


def verbs2basicform(words):
    ret = []
    for w in words:
        analysis = wn.synsets(w)
        if any([a.pos() == 'v' for a in analysis]):
            w = WordNetLemmatizer().lemmatize(w, 'v')
        ret.append(w)
    return ret

