
from glob import glob
import argparse
import math
import os
import shutil
import zipfile

parser = argparse.ArgumentParser(description='Extract and segment data into training and testing')
parser.add_argument('-N', type=float, help='percentage of training data (default: 0.8)')
args = parser.parse_args()

N = 0.8 if args.N is None else args.N

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DATA_ZIP = os.path.join(DIR_PATH, 'asl-alphabet.zip')
TRAIN_ZIP = os.path.join(DIR_PATH, 'asl_alphabet_train.zip')
TEST_ZIP = os.path.join(DIR_PATH, 'asl_alphabet_test.zip')
DATA_DIR = os.path.join(DIR_PATH, 'data')
INTERMEDIATE_DIR = os.path.join(DATA_DIR, 'asl_alphabet_train')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

def removeIfExists(f):
    if os.path.exists(f):
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)

print 'removing old files.........'
removeIfExists(DATA_DIR)
removeIfExists(TRAIN_ZIP)
removeIfExists(TEST_ZIP)

print 'extracting zip data........'
zip_ref = zipfile.ZipFile(DATA_ZIP, 'r')
zip_ref.extractall('.')
zip_ref.close()

print "extracting training data..."
zip_ref = zipfile.ZipFile(TRAIN_ZIP, 'r')
zip_ref.extractall(DATA_DIR)
zip_ref.close()

print "moving training dir........"
shutil.move(INTERMEDIATE_DIR, TRAIN_DIR)

print "moving test data..........."
os.makedirs(TEST_DIR)
training_classes = glob(TRAIN_DIR + '/*')
for training_class in training_classes:
    test_class_dir = training_class.replace('train', 'test')
    os.makedirs(test_class_dir)
    class_files = glob(training_class + '/*')
    class_files.sort()
    bound = int(N * len(class_files))
    for training_img in class_files[bound:]:
        testing_img = training_img.replace('train', 'test')
        shutil.move(training_img, testing_img)

print "cleaning up zips..........."
removeIfExists(TRAIN_ZIP)
removeIfExists(TEST_ZIP)
