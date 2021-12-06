"""
Obtaining E-ANEW and Subjectivity-clues, and merging them into final lexicon
:author: Qizhi Li
"""
from static_data import file_path as fp
from utils import Utils


def read_subjectivity_clues():
    """
    Loading the data of subjectivity_clues:
        1. if polarity is positive, then the orientation is 1;
        2. if polarity is negative, then the orientation is 2;
        3. if polarity is neutral, then the orientation is 0;
        4. we discard others situation.
    :return sc_lexicon: dict
            subjectivity_clues lexicon
    """
    sc_lexicon = {}
    with open(fp.Subjectivity_clues_lexicon) as f:
        for line in f:
            entry = line.split(' ')
            word = entry[2].split('=')[1]
            polarity = entry[len(entry) - 1].split('=')[1].strip()

            if polarity == 'positive':
                orientation = 1
            elif polarity == 'negative':
                orientation = 2
            elif polarity == 'neutral':
                orientation = 0
            else:
                continue

            sc_lexicon[word] = orientation

    return sc_lexicon


def read_ANEW():
    """
    Loading E-ANEW data, we only load the third column
        1. if the data in the third column is 5, then the orientation is 0;
        2. if the data in the third column is in the range of [1, 5),
           then the orientation is 2;
        3. if the data in the third column is in the range of (5, 9],
           then the orientation is 1;
    :return ANEW_lexicon: dict
            E-ANEW lexicon
    """
    ANEW_lexicon = {}
    with open(fp.EANEW_lexicon) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
            else:
                entry = line.split(',')
                if float(entry[2]) == 5:
                    ANEW_lexicon[entry[1]] = 0
                elif float(entry[2]) > 5:
                    ANEW_lexicon[entry[1]] = 1
                else:
                    ANEW_lexicon[entry[1]] = 2

    return ANEW_lexicon


def generate_final_lexicon(ANEW_lexicon, sc_lexicon):
    """
    Merging the subjectivity-clues and E-ANEW.
        1. if word i appears in only one lexicon,
           we directly add it to the final lexicon;
        2. if word i has the same sentiment orientation in the two lexicon,
           we directly add it to the final lexicon;
        3. if the sentiment orientation of word i is neutral in one lexicon,
           but the sentiment orientation is not neutral in another lexicon,
           we add the word with non-neutral sentiment orientation to the final lexicon;
        4. if word $i$ has opposite sentiment orientations (positive and negative)
           in the two lexicon, we will discard the word.
    :param ANEW_lexicon: dict
            ANEW lexicon
    :param sc_lexicon: dict
            subjectivity_clues lexicon
    :return final_lexicon: dict
    """
    final_lexicon = {}
    not_common = 0
    common = 0
    abandon = 0
    for k in ANEW_lexicon.keys():
        if k not in sc_lexicon.keys():
            final_lexicon[k] = ANEW_lexicon[k]
            not_common += 1
        elif ANEW_lexicon[k] == sc_lexicon[k]:
            final_lexicon[k] = ANEW_lexicon[k]
            common += 1
        elif ANEW_lexicon[k] * sc_lexicon[k] == 0:
            final_lexicon[k] = ANEW_lexicon[k] if ANEW_lexicon[k] != 0 else sc_lexicon[k]
            common += 1
        else:
            abandon += 1
            continue

    for k in sc_lexicon.keys():
        if k not in final_lexicon.keys():
            final_lexicon[k] = sc_lexicon[k]
            not_common += 1

    print('The final lexicon size = %d' % len(final_lexicon))
    print('The number of common words = %d' % common)
    print('The number of words in one but not in another lexicon = %d' % not_common)
    print('The number of abandon words = %d' % abandon)

    return final_lexicon


def save_final_lexicon(final_lexicon, utils):
    """
    Saving the final lexicon
    :param final_lexicon: dict
    :param utils: Object
    """
    utils.write_file('json', fp.final_lexicon, final_lexicon)


utils = Utils()
final_lexicon = generate_final_lexicon(ANEW_lexicon=read_ANEW(),
                                       sc_lexicon=read_subjectivity_clues())
save_final_lexicon(final_lexicon, utils)