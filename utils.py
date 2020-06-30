import csv, string
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict

START_WORD = 'xstart'
DELIM = '\n'

N_EMBEDDINGS = 50 # must be 25, 50, 100, or 200
N_HIDDEN = 100

BATCH_SIZE = 32

### functions for parsing data ###

def get_glove_vectors(vocab):
    filename = 'glove.' + str(N_EMBEDDINGS) + 'd.txt'
    print('reading glove file...')
    glove_vectors = {}
    for line in open(filename, encoding='utf8'):
        line = line.strip().split()
        glove_vectors[line[0]] = [float(e) for e in line[1:]]
    print('done.')
    result = []
    for word in vocab:
        if word.lower() in glove_vectors:
            result.append(np.array(glove_vectors[word.lower()]))
        else:
            result.append(np.random.uniform(-3, 3, [N_EMBEDDINGS]))
    return np.array(result)

def get_data():
    data = _deliminate_data(_get_raw_data())

    result = []
    for haiku in data:
        to_add = True
        for word in haiku:
            if not (word == DELIM or word == START_WORD):
                if isinstance(nsyl(word), bool):
                    to_add = False
        if to_add:
            result.append(haiku)
        
    return result

def _deliminate_data(data):
    result = []
    for poem in data:
        curr_haiku = [START_WORD]
        for line in poem:
            curr_haiku += line + [DELIM]
        result.append(curr_haiku)
    return result

def _get_raw_data():
    # reading the poetRNN data 
    result = []
    with open('haikus.csv') as haikus:
        readCSV = csv.reader(haikus, delimiter='\"')
        for row in readCSV:
            haiku = row[0].split(DELIM)[:-1]
            if len(haiku) == 3:
                haiku = [word_tokenize(l) for l in haiku if not '?' in l]
                if len(haiku) == 3:
                    result.append(haiku)

    # reading the haikuzao data
    with open('haiku.txt') as haikus:
        curr_haiku = []
        for line in haikus:
            if line == DELIM:
                if len(curr_haiku) == 3:
                    result.append(curr_haiku)
                curr_haiku = []
            else:
                curr_haiku.append(word_tokenize(line))
    return result

### functions for working with poems ###

def poem_is_haiku(poem):
    line1, line2, line3 = _poem_to_lines(poem)
    if 5 == nsyl_line(line1) and 7 == nsyl_line(line2) and 5 == nsyl_line(line3):
        return True
    else:
        return False

def print_poem(poem):
    d = TreebankWordDetokenizer()

    line1, line2, line3 = _poem_to_lines(poem)
    n1 = nsyl_line(line1)
    n2 = nsyl_line(line2)
    n3 = nsyl_line(line3)
    line1 = d.detokenize(line1)
    line2 = d.detokenize(line2)
    line3 = d.detokenize(line3)
    print(str(n1) + ' | ' + line1 + '\n' + \
          str(n2) + ' | ' + line2 + '\n' + \
          str(n3) + ' | ' + line3 + '\n')

def _poem_to_lines(poem):
    delims = [i for i, x in enumerate(poem) if x == DELIM]
    
    # one line poem
    if len(delims) == 0:
        return (poem[1:], [], [])
    # two line poem
    if len(delims) == 1:
        line1 = poem[1:delims[0]]
        line2 = poem[delims[0]:]
        if line2[-1] == DELIM:
            line2 = line2[:-1]
        return (line1, line2, [])
    # three line poem
    line1 = poem[1:delims[0]]
    line2 = poem[delims[0]+1:delims[1]]
    line3 = poem[delims[1]+1:]

    if len(delims) == 3:
        line3 = line3[:-1]

    return (line1, line2, line3)

### functions for counting syllables ###
import nltk
d = cmudict.dict()

def nsyl(word):
    try:
        n = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
        if word == '\'s':
            return 0
    except KeyError:
        # if word not found in cmudict
        n = 0
        if not word in string.punctuation and not word == START_WORD and not word == DELIM:
            return False
    return n 

def nsyl_line(line):
    n = 0
    for word in line:
        n += nsyl(word)
    return n

### functions for pre-processing data ###
def remove_duplicates_glove(vocab):
    filename = 'glove.twitter.27B.' + str(N_EMBEDDINGS) + 'd.txt'
    words = [w.lower() for w in vocab]
    with open('glove.' + str(N_EMBEDDINGS) + 'd.txt', 'w+') as f:
        for line in open(filename, encoding='utf8'):
            if line.strip().split()[0] in words:
                f.write(line)
            