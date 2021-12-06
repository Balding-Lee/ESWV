"""
获得数据的最大长度
:author: Qizhi Li
"""
import matplotlib.pyplot as plt

from preprocess import load_experiment_dataset


def get_max_seq_length(type_='semeval'):
    """
    Obtaining the max sequence length of the dataset.
    :param type_: str
            'semeval'
            'SST'
            default: 'semeval'
    :return len_frequency: dict
            {0: f1, 10: f2, 20: f3, ...}
    """
    len_frequency = {}
    X_train, _, X_dev, _, X_test, _ = load_experiment_dataset.load_data(type_, False)

    all_data = []
    all_data.extend(X_train)
    all_data.extend(X_dev)
    all_data.extend(X_test)

    for sentence in all_data:
        length = int(len(sentence) / 10) * 10  # The interval is 10
        try:
            len_frequency[length] += 1
        except KeyError:
            len_frequency[length] = 1

    return len_frequency


def draw_fig(x, y):
    """
    :param x: list
            interval
    :param y: list
            frequency
    """
    plt.bar(x, y)
    plt.xlabel('The length of sentences')
    plt.ylabel('The frequency of each length')
    plt.show()


len_frequency = get_max_seq_length(type_='semeval')
draw_fig(list(len_frequency.keys()), list(len_frequency.values()))