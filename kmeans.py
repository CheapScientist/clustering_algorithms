import numpy as np
import imageio
from matplotlib import pyplot as plt
import sys
import os
import time
import math
import random


# class k_means:
#
#     def __init__(self):
#         self.pixels = np.zeros(1)
#         self.MAX_ITERATION = 10
#         self.K = 5
#
#     # randomly select K points as centers
#     def generate_random_centers(self):
#         return np.random.rand(self.K, 3)*255
#
#     # get distance
#     def get_dist(self, pixel1, pixel2):
#         return np.sum(np.square(pixel1 - pixel2), axis=-1)
#
#     # get class assignment for one pixel
#     def class_assignment(self, pixel, centers):
#         dist = self.get_dist(pixel, centers)
#         return np.argmin(dist)
#
#     # get class assignments
#     def get_class_assignments(self, centers):
#         classes = np.zeros((len(self.pixels), 1), dtype = int)
#         # start = time.time()
#         for i in range(len(self.pixels)):
#             classes[i] = self.class_assignment(self.pixels[i], centers)
#         # t = time.time() - start
#         # print(f'time taken: {t}')
#         return classes


def mykmeans(pixels, K):
    # I've tried different ways to do kmeans and found below is faster + I had no time to debug object-oriented style

    # randomly select K centers
    r, c, _ = np.shape(pixels)
    K = min(K, r*c)
    im = pixels.reshape(r * c, 3)
    seen, cnt = set(), 0
    # generate random data points as centers, seen makes sure there's no duplicates
    centers = np.asarray([[0] * 3 for _ in range(K)])
    while cnt < K:
        rr, cc = math.floor(r * random.random()), math.floor(c * random.random())
        if (rr, cc) in seen:
            continue
        seen.add((rr, cc))
        centers[cnt] = pixels[rr][cc]
        cnt += 1

    MAX_ITERATION = 50
    EPS = 1e-4
    l = r * c

    classes = np.asarray([0] * l)
    # start iteration
    for iteration in range(MAX_ITERATION):
        flag = True
        idx = 0
        tmp = np.zeros((K, 3))
        tmp_cnt = [0] * K
        for x in range(r):
            for y in range(c):
                cur_pixel = pixels[x][y]
                min_dist = float('inf')
                for i, center_pixel in enumerate(centers):
                    cur_dist = sum([(cur_pixel[i] - center_pixel[i]) ** 2 for i in range(3)])
                    if min_dist > cur_dist:
                        min_dist = cur_dist
                        classes[idx] = i
                cur_assign = classes[idx]
                tmp_cnt[cur_assign] += 1
                for j in range(3):
                    tmp[cur_assign][j] += cur_pixel[j]
                idx += 1

        for x in range(K):
            # if there is an empty cluster:
            if tmp_cnt[x] == 0:
                flag = False
                break
            for y in range(3):
                tmp[x][y] = tmp[x][y] / tmp_cnt[x]
        if np.max(abs(tmp - centers)) < EPS or not flag:
            break
        for x in range(K):
            for y in range(3):
                centers[x][y] = tmp[x][y]
        print('k means, iteration: ', iteration)

    return classes, centers


    #
    # r, c, _ = pixels.shape
    # EPS = 1e-6
    # K = min(K, r * c)
    # l = r * c
    # # if K is too large, set K = R * C
    # # preprocess data
    # pixels = pixels.reshape(r * c, 3).astype(np.float32)
    # # initialize class
    # my_kmeans = k_means()
    # my_kmeans.K = K
    # my_kmeans.pixels = pixels
    # centers = my_kmeans.generate_random_centers()
    # classes = my_kmeans.get_class_assignments(centers)
    #
    # for iteration in range(my_kmeans.MAX_ITERATION):
    #     # tried a few optimizations, writing function inside makes it faster
    #     flag = True
    #     idx = 0
    #     tmp = np.zeros((K, 3))
    #     tmp_cnt = [0] * K
    #     for x in range(r):
    #         cur_pixel = my_kmeans.pixels[x]
    #         min_dist = float('inf')
    #         for i, center_pixel in enumerate(centers):
    #             cur_dist = my_kmeans.get_dist(cur_pixel, center_pixel)
    #             if min_dist > cur_dist:
    #                 min_dist = cur_dist
    #                 classes[idx] = i
    #         cur_assign = classes[idx][0]
    #         tmp_cnt[cur_assign] += 1
    #         for j in range(3):
    #             tmp[cur_assign][j] += cur_pixel[j]
    #         idx += 1
    #
    #     for x in range(K):
    #         if tmp_cnt[x] == 0:
    #             flag = False
    #         for y in range(3):
    #             tmp[x][y] = tmp[x][y] / tmp_cnt[x]
    #     if np.max(abs(tmp - centers)) < EPS or not flag:
    #         break
    #     centers = tmp.copy()
    # classes = my_kmeans.get_class_assignments(centers)
    # return classes, centers


class k_medoids:

    # initialization
    def __init__(self):
        self.pixels = np.zeros(1)
        self.MAX_ITERATION = 1000
        self.K = 5

    # randomly select K data points as centers
    def generate_random_centers(self):
        idx = np.random.choice(self.pixels.shape[0], self.K)
        return self.pixels[idx]

    # get hamming_dist with two pixels
    def get_dist(self, pixel1, pixel2):
        return np.sum(np.square(pixel1 - pixel2), axis=-1)

    # get class assignment for one pixel
    def class_assignment(self, pixel, centers):
        dist = self.get_dist(pixel, centers)
        return np.argmin(dist)

    # get class assignments
    def get_class_assignments(self, centers):
        classes = np.zeros((len(self.pixels), 1), dtype=int)
        # start = time.time()
        for i in range(len(self.pixels)):
            classes[i] = self.class_assignment(self.pixels[i], centers)
        # t = time.time() - start
        # print(f'time taken: {t}')
        return classes

    # calculate what the cost is
    def calculate_cost(self, classes, centers):
        # start = time.time()
        res = 0
        n = len(self.pixels)
        for i in range(n):
            res += self.get_dist(self.pixels[i], centers[classes[i]])
        # t = time.time() - start
        # print(f'time taken: {t}')
        return res


def mykmedoids(pixels, K):
    r, c, _ = pixels.shape
    # if K is too large, set K = R * C
    K = min(K, r * c)
    # preprocess data
    pixels = pixels.reshape(r * c, 3).astype(np.float32)
    # assign class
    my_k_medoids = k_medoids()
    my_k_medoids.K, my_k_medoids.pixels = K, pixels
    # initialize centers, classes and cost
    centers = my_k_medoids.generate_random_centers()
    classes = my_k_medoids.get_class_assignments(centers)
    cost = my_k_medoids.calculate_cost(classes, centers)

    for i in range(my_k_medoids.MAX_ITERATION):
        # new_centers = my_k_medoids.generate_random_centers()
        flag = False
        # PAM algorithm, greedily swap centers, if the cost decreases throughout one loop, continue the loop,
        # otherwise exit
        for j in range(my_k_medoids.K):
            which_center = my_k_medoids.pixels[classes[:, 0] == j]
            new_centers = centers.copy()
            # if nothing is assigned to this center, continue
            if len(which_center) == 0:
                continue
            new_centers[j] = pixels[np.random.choice(which_center.shape[0], 1)]
            new_classes = my_k_medoids.get_class_assignments(new_centers)
            new_cost = my_k_medoids.calculate_cost(new_classes, new_centers)
            if new_cost < cost:
                print('cost updated, the new cost is: ', new_cost, 'iteration: ', i)
                cost = new_cost
                centers = new_centers.copy()
                flag = True
            else:
                print('cost not updated, continue with previous cost: ', cost, 'iteration: ', i)
            # continue
        if not flag:
            break

    classes = my_k_medoids.get_class_assignments(centers)
    return classes, centers


def main():
    if (len(sys.argv) < 2):
        print("Please supply an image file")
        return

    image_file_name = sys.argv[1]
    K = 5 if len(sys.argv) == 2 else int(sys.argv[2])
    print(image_file_name, K)
    im = np.asarray(imageio.imread(image_file_name))

    fig, axs = plt.subplots(1, 2)

    classes, centers = mykmedoids(im, K)
    print(classes, centers)
    new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
    imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) +
                    os.path.splitext(image_file_name)[1], new_im)
    axs[0].imshow(new_im)
    axs[0].set_title('K-medoids')

    classes, centers = mykmeans(im, K)
    print(classes, centers)
    new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
    imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) +
                    os.path.splitext(image_file_name)[1], new_im)
    axs[1].imshow(new_im)
    axs[1].set_title('K-means')

    plt.show()


if __name__ == '__main__':
    main()
