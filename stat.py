import numpy as np

if __name__ == '__main__':
    stat = np.load('stat.npy')
    length = len(stat)

    for s in stat:
        print('%d/%d: d_loss: %.4f,  a_loss: %.4f' % (s[0], length, s[1], s[2]))
