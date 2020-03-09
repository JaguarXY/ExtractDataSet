# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 17:26
# @File    : getMnist.py
# @Version : Python 2.7.10
# @Author  : XYJ
import time,os,struct
import shutil
import PIL.Image
import numpy as np
# set the output path
outdir = 'D:\DeepLearning\Caffe_DIGIST\DataSets\Mnist_ok'
# use *.png format
file_extension = 'png'

def processData_MNIST():
    __extract_images('train-images.bin', 'train-labels.bin', 'train')
    __extract_images('test-images.bin', 'test-labels.bin', 'test')

def __extract_images(images_file, labels_file, phase):
    """
    Extract information from binary files and store them as images
    """
    labels = __readLabels(os.path.join(outdir, labels_file))
    images = __readImages(os.path.join(outdir, images_file))
    assert len(labels) == len(images), '%d != %d' % (len(labels), len(images))

    output_dir = os.path.join(outdir, phase)
    mkdir_safely(output_dir, clean=True)
    with open(os.path.join(output_dir, 'labels.txt'), 'w') as outfile:
        for label in xrange(10):
            outfile.write('%s\n' % label)
    with open(os.path.join(output_dir, '%s.txt' % phase), 'w') as outfile:
        for index, image in enumerate(images):
            dirname = os.path.join(output_dir, labels[index])
            mkdir_safely(dirname)
            filename = os.path.join(dirname, '%05d.%s' % (index, file_extension))
            image.save(filename)
            outfile.write('%s %s\n' % (filename, labels[index]))


def __readLabels(filename):
    """
    Returns a list of ints
    """
    print 'Reading labels from %s ...' % filename
    labels = []
    with open(filename, 'rb') as infile:
        infile.read(4)  # ignore magic number
        count = struct.unpack('>i', infile.read(4))[0]
        data = infile.read(count)
        for byte in data:
            label = struct.unpack('>B', byte)[0]
            labels.append(str(label))
    return labels


def __readImages(filename):
    """
    Returns a list of PIL.Image objects
    """
    print 'Reading images from %s ...' % filename
    images = []
    with open(filename, 'rb') as infile:
        infile.read(4)  # ignore magic number
        count = struct.unpack('>i', infile.read(4))[0]
        rows = struct.unpack('>i', infile.read(4))[0]
        columns = struct.unpack('>i', infile.read(4))[0]

        for i in xrange(count):
            data = infile.read(rows * columns)
            image = np.fromstring(data, dtype=np.uint8)
            image = image.reshape((rows, columns))
            # image = 255 - image  # now black digit on white background
            images.append(PIL.Image.fromarray(image))
    return images
# A safely approach of making dir
def mkdir_safely(d, clean=False):
    if os.path.exists(d):
        if clean:
            shutil.rmtree(d)
        else:
            return
    os.mkdir(d)
if __name__ == "__main__":
    start = time.time()
    print 'Extract Start!'
    if os.path.exists(outdir + '\\finish'):
        print 'File Extracting has already done.'
    else:
        processData_MNIST()
        mkdir_safely(outdir + '\\finish')
    print 'Done after %s secends.' % (time.time() - start)