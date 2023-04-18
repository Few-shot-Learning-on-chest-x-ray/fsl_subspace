import os
import time
import pprint
import torch
from sklearn.metrics import confusion_matrix

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)

    acc = [0 for c in range(3)]  
    for c in range(3):
        acc[c] = (pred.eq(label) * label.eq(c)).float() / max((label.eq(c)).sum(), 1)


    matrix = confusion_matrix(label.cpu().detach().numpy(), pred.cpu().detach().numpy())
    pca = matrix.diagonal()/matrix.sum(axis=1)

    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy, pca * 100
    
class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)