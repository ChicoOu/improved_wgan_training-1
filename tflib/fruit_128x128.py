import numpy as np
import scipy.misc
import time
import os

image_size = 128
test_size = 50


def make_generator(data_dir, n_files, batch_size):
    epoch_count = [1]

    def get_epoch():
        images = np.zeros((batch_size, 3, image_size,
                           image_size), dtype='int32')
        files = [name for name in os.listdir(data_dir)
                 if os.path.isfile(os.path.join(data_dir, name))]

        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, name in enumerate(files):
            image = scipy.misc.imread("{}/{}".format(data_dir, name))
            images[n % batch_size] = image.transpose(2, 0, 1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch


def make_testset(test_dir):
    images = np.zeros((test_size, 3, image_size, image_size), dtype=np.int32)
    files = [name for name in os.listdir(test_dir)
             if os.path.isfile(os.path.join(test_dir, name))]
    for n, name in enumerate(files):
        image = scipy.misc.imread("{}/{}".format(test_dir, name))
        images[n] = image.transpose(2, 0, 1)
    return images


def load(batch_size, data_dir='./datasets/002orange/training', test_dir='./datasets/002orange/test'):
    if not os.path.isdir(data_dir):
        raise Exception("{} is not a directory".format(data_dir))
    file_count = 202599
    print('load {} files'.format(file_count))
    return make_generator(data_dir, file_count, batch_size), make_testset(test_dir)


if __name__ == '__main__':
    train_gen, test_images = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print("{}\t{}".format(str(time.time() - t0), batch[0][0, 0, 0, 0]))
        if i == 5:
            print(batch[0].shape, batch[0].dtype)
            break
        t0 = time.time()
