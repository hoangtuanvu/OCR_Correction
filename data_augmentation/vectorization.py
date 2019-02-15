from data_augmentation.constants import INVERT, ADD_NOISE_TO_DATA, NOISE_PROBS, OUT_SRC_PATH, OUT_TAR_PATH
from data_augmentation.generating_noise import add_noise_to_data, build_vocabulary
from data_augmentation.generating_data import LazyDataLoader

print('Vectorization...')

DATA_LOADER = LazyDataLoader()
VOCAB = build_vocabulary()

INPUT_MAX_LEN, OUTPUT_MAX_LEN, TRAINING_SIZE = DATA_LOADER.statistics()

print('Generating data...')
count = 0

f_src = open(OUT_SRC_PATH, 'w')
f_tar = open(OUT_TAR_PATH, 'w')

while count < TRAINING_SIZE:
    count += 1
    x, y = DATA_LOADER.next()
    # Pad the data with spaces such that it is always MAXLEN.
    q = x
    query = q
    ans = y

    if ADD_NOISE_TO_DATA:
        # print('Old query =', query, end='  |   ')
        query, _ = add_noise_to_data(input_str=query, probs=NOISE_PROBS, vocabulary=VOCAB)
        # print('Query =', query, '  |   Noise type =', noise_type)

    if INVERT:
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
        # space used for padding.)
        query = query[::-1]

    f_src.write(query + '\n')
    f_tar.write(ans + '\n')

    if count % 10000 == 0:
        print('Processed: {}'.format(count))
