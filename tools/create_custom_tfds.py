import os 
import tensorflow as tf 
from PIL import Image  


cwd = 'D:/Google Cloud (60747050S)/Research/trajectory-data/'
train_record_path = "D:/Google Cloud (60747050S)/Research/trajectory-data/output/trajectory_train.tfrecords"
test_record_path = "D:/Google Cloud (60747050S)/Research/trajectory-data/output/trajectory_test.tfrecords"

classes={'novel_tank','mirror_biting'}

def _bytes_features(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _int64_features(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def create_train_record():
    writer = tf.io.TFRecordWriter("trajectory_train.tfrecords")

    NUM = 1  # represent build process (counting)
    for index, name in enumerate(classes):
        class_path = cwd + name + '/'
        l = int(len(os.listdir(class_path)) * 0.7)  # 70% for test dataset

        for img_name in os.listdir(class_path)[:l]: 
            img_path = class_path + img_name  # 
            img = Image.open(img_path)
            img = img.resize((128,128))  # resize the size of picture
            img_raw = img.tobytes()  # Turn image into bytes
            example = tf.train.Example(
                features = tf.train.Features(feature={
                    "label": tf.train.Feature(_int64_features(index)),
                    'img_raw': tf.train.Feature(_bytes_features(img_raw))
            }))  # Package into Example
            writer.write(example.SerializeToString())
            print('Creating train record in ', NUM)
            NUM += 1
    writer.close()
    print("Create train_record successful!")

def create_test_record():
    writer = tf.python_io.TFRecordWriter(test_record_path)
    NUM = 1
    for index, name in enumerate(classes):
        class_path = cwd + '/' + name + '/'
        l = int(len(os.listdir(class_path)) * 0.7)
        for img_name in os.listdir(class_path)[l:]:  # 30% as test dataset
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((128, 128))
            img_raw = img.tobytes()  # Turn image into bytes
            # print(index,img_raw)
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label":_int64list(index),
                    'img_raw':_byteslist(img_raw)
                }))
            writer.write(example.SerializeToString())
            print('Creating test record in ',NUM)
            NUM += 1
    writer.close()
    print("Create test_record successful!")


def read_record(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [128, 128, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # Generalize
    label = tf.cast(label, tf.int32)
    return img, label
    

def get_batch_record(filename, batch_size):
    image,label = read_record(filename)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                     batch_size = batch_size,
                                                     capacity = 2000,
                                                     min_after_dequeue = 1000)
    return image_batch, label_batch


def main():
    create_train_record()
    create_test_record()
if __name__ == '__main__':
    main()