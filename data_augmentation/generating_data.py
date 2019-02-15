from data_augmentation.constants import SOURCE_PATH, TARGET_PATH


def stream_from_file():
    """
    Load dataset
    """
    with open(SOURCE_PATH, 'r') as f1:
        with open(TARGET_PATH, 'r') as f2:
            for (sent1, sent2) in zip(f1, f2):
                yield sent1.strip(), sent2.strip()


class LazyDataLoader:
    def __init__(self):
        self.stream = stream_from_file()

    def next(self):
        try:
            return next(self.stream)
        except:
            self.stream = stream_from_file()
            return self.next()

    def statistics(self):
        max_len_value_x = 0
        max_len_value_y = 0
        num_lines = 0
        self.stream = stream_from_file()
        for x, y in self.stream:
            max_len_value_x = max(max_len_value_x, len(x))
            max_len_value_y = max(max_len_value_y, len(y))
            num_lines += 1

        print('max_len_value_x =', max_len_value_x)
        print('max_len_value_y =', max_len_value_y)
        print('num_lines =', num_lines)
        return max_len_value_x, max_len_value_y, num_lines


if __name__ == '__main__':
    # how to use it.
    ldl = LazyDataLoader()
    print(ldl.statistics())
