import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch


def read_idx3_ubyte_pixel_file(file_path):
    print("Info : begin to load id3 ubyte pixel file " + file_path)
    with open(file_path, 'rb') as f:
        pixel_file = f.read()
        offset = 0
        magic_offset = 4
        magic_number_bytes = pixel_file[: offset + magic_offset]
        magic_number = int.from_bytes(magic_number_bytes, byteorder='big')
        print("Info : magic number " + str(magic_number))
        offset = offset + magic_offset

        image_num_offset = 4
        image_num_bytes = pixel_file[offset: offset + image_num_offset]
        image_num = int.from_bytes(image_num_bytes, byteorder='big')
        print("Info : image number " + str(image_num))
        offset = offset + image_num_offset

        height_offset = 4
        height_bytes = pixel_file[offset: offset + height_offset]
        height = int.from_bytes(height_bytes, byteorder='big')
        print("Info : image height " + str(height))
        offset = offset + height_offset

        width_offset = 4
        width_bytes = pixel_file[offset: offset + width_offset]
        width = int.from_bytes(width_bytes, byteorder='big')
        print("Info : image width" + str(width))
        offset = offset + width_offset

        picture_offset = height * width
        image_list = []
        for i in range(image_num):
            if i % 1000 == 0:
                print("Info : load picture " + str(i))
            image = []
            for j in range(picture_offset):
                pixel = int.from_bytes(pixel_file[offset:offset + 1], byteorder='big')
                offset = offset + 1
                image.append(pixel / 255)
            # print("Info : picture " + str(i))
            # print("Info : picture size " + str(len(image)))
            # print("Info : picture pixels " + str(image))

            image_list.append(np.array(image, dtype=float).reshape(28, 28))
            # cv2.imwrite("./Resources/PNGs/img_" + str(i) + ".png", np.array(image).reshape(28, 28))
    return image_list, (height, width)


def read_idx3_ubyte_label_file(file_path):
    print("Info : begin to load id3 ubyte label file " + file_path)
    with open(file_path, 'rb') as f:
        label_file = f.read()
        offset = 0
        magic_offset = 4
        magic_number_bytes = label_file[: offset + magic_offset]
        magic_number = int.from_bytes(magic_number_bytes, byteorder='big')
        print("Info : magic number " + str(magic_number))
        offset = offset + magic_offset

        image_num_offset = 4
        image_num_bytes = label_file[offset: offset + image_num_offset]
        image_num = int.from_bytes(image_num_bytes, byteorder='big')
        print("Info : image number " + str(image_num))
        offset = offset + image_num_offset

        label_list = []
        for i in range(image_num):
            if i % 10000 == 0:
                print("Info : load label " + str(i))
            label_bytes = label_file[offset: offset + 1]
            offset = offset + 1
            label = int.from_bytes(label_bytes, 'big')
            label_list.append(label)
            # print("Info : picture " + str(i) + " is number " + str(label))
        print("Info : picture label " + str(label_list))
    return label_list
    
    
if __name__ == '__main__':
    read_idx3_ubyte_pixel_file('Resources/train-images.idx3-ubyte')
    read_idx3_ubyte_label_file('Resources/train-labels.idx1-ubyte')
