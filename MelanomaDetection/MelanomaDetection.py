import numpy as np
import scipy.stats as sci
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import os
import sys
import math as ma
import copy as cp


np.set_printoptions(threshold=sys.maxsize)


#%% importing jpg images into list of them:
def load_images(folder="data"):
    images = []
    for filename in sorted(os.listdir(folder)):
        # Each element img is a tuple composed with : (image,
        #                                              name of the image
        #                                              number of cluster to use,
        #                                              size of the half of the clean matrix,
        #                                              size of the half of the border matrix)

        # Apply specific value to the value of the tuple depending on the image
        # print(filename)
        if filename == "low_risk_01.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 5, 2, 4)
        elif filename == "low_risk_04.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 4, 4, 4)
        elif filename == "low_risk_05.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 3, 2, 3)
        elif filename == "low_risk_07.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 4, 2, 4)
        elif filename == "low_risk_10.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 4, 2, 4)

        elif filename == "medium_risk_01.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 4, 2, 4)
        elif filename == "medium_risk_05.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 4, 3, 4)
        elif filename == "medium_risk_08.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 3, 3, 7)
        elif filename == "medium_risk_09.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 3, 3, 5)
        elif filename == "medium_risk_10.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 2, 3, 5)
        elif filename == "medium_risk_11.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 3, 3, 6)

        elif filename == "melanoma_01.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 3, 3, 4)
        elif filename == "melanoma_03.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 2, 3, 5)
        elif filename == "melanoma_11.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 5, 2, 4)
        elif filename == "melanoma_12.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 3, 2, 6)
        elif filename == "melanoma_13.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 3, 2, 5)
        elif filename == "melanoma_16.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 3, 3, 7)
        elif filename == "melanoma_19.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 4, 3, 5)
        elif filename == "melanoma_20.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 3, 3, 7)
        elif filename == "melanoma_21.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 4, 3, 7)
        elif filename == "melanoma_22.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 3, 3, 6)
        elif filename == "melanoma_25.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 3, 3, 5)
        elif filename == "melanoma_26.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 5, 3, 10)
        elif filename == "melanoma_27.jpg":
            img = (mpimg.imread(os.path.join(folder, filename)), 8, 2, 4)

        else:
            img = (mpimg.imread(os.path.join(folder, filename)), 3, 2, 4)

        if img is not None:
            images.append(img)
    return images


#%% displaying the image
def show_image(work_image, title):
    plt.figure
    plt.imshow(work_image)
    plt.title(title)
    plt.show()


#%% using K-means to distinguish main colours


def quantize_picture(image, nb_of_clusters=3):

    kmeans = KMeans(n_clusters=nb_of_clusters, random_state=0)
    # dimension are (L,l,colour) where colour can be Red, Green or Blue
    [N1, N2, N3] = image.shape
    # new dimension is ((L*l),colour)
    images_2D = image.reshape((N1 * N2), N3)
    kmeans.fit(images_2D)

    # cast the obtained values en generate the centroids tab
    centroids = kmeans.cluster_centers_.astype("uint8")
    label = kmeans.labels_
    # quantized_image = centroids[label].reshape(N1, N2, N3)

    # find the centroid corresponding to the darkest pixels
    smallest_val = 255
    for i in range(nb_of_clusters):
        for j in range(3):
            if centroids[i, j] < smallest_val:
                smallest_val = centroids[i, j]
                # darkest_centroid = centroids[i]
                index_dark = i

    # Generate the image with only two colours
    work_image = kmeans.labels_.reshape(N1, N2) == index_dark
    work_image = work_image * 1
    return work_image


#%% This function aims to resizing the original image closer the mole
def resize_picture(work_image, margin_height=30, margin_width=30, half_matrix_size=2):
    [N1, N2] = work_image.shape

    # Clear 50 pixel from each border to remove possible shadow
    for i in range(N1):
        for j in range(50):
            work_image[i, j] = 0
            work_image[i, N2 - 1 - j] = 0

    for i in range(N2):
        for j in range(50):
            work_image[j, i] = 0
            work_image[N1 - 1 - j, i] = 0

    up_border = best_border_coord_up(
        work_image, margin_height=margin_height, half_matrix_size=half_matrix_size
    )
    # print(up_border)
    down_border = best_border_coord_down(
        work_image, margin_height=margin_height, half_matrix_size=half_matrix_size
    )
    # print(down_border)
    left_border = best_border_coord_left(
        work_image, margin_width=margin_width, half_matrix_size=half_matrix_size
    )
    # print(left_border)
    right_border = best_border_coord_right(
        work_image, margin_width=margin_width, half_matrix_size=half_matrix_size
    )
    # print(right_border)

    new_work_image = work_image[
        up_border[0] : down_border[0], left_border[1] : right_border[1]
    ]
    return new_work_image


#%% defining a new up border of the image
def best_border_coord_up(work_image, margin_height=30, half_matrix_size=2):
    # Define the best poistion to estimate the border (for the other coordinate)
    [N1, N2] = work_image.shape

    matrix_size = 2 * half_matrix_size + 1

    distances = np.zeros((N2 - half_matrix_size, 2), dtype=int)
    ones_matrix = np.ones((matrix_size, matrix_size))

    for i in range(half_matrix_size, N2 - half_matrix_size):
        # print("i : "+str(i))
        done = False
        dist = half_matrix_size
        distances[i - half_matrix_size, 0] = i

        while done == False:

            around_dist = np.zeros((matrix_size, matrix_size), dtype=int)
            for n in range(i - half_matrix_size, i + half_matrix_size + 1):
                small_matrix_index_n = n % matrix_size
                for m in range(dist - half_matrix_size, dist + half_matrix_size + 1):
                    small_matrix_index_m = m % matrix_size
                    around_dist[
                        small_matrix_index_m, small_matrix_index_n
                    ] = work_image[m, n]

            if np.array_equal(around_dist, ones_matrix) or dist == N1 - (
                half_matrix_size + 1
            ):
                done = True
            else:
                dist = dist + 1
        distances[i - half_matrix_size, 1] = dist

    distances = np.delete(distances, len(distances) - 1, 0)
    distances = np.delete(distances, len(distances) - 1, 0)

    # print(distances)
    opti_coord = np.argmin(distances, axis=0)
    opti_up = distances[opti_coord[1], 0]
    # print(opti_up)
    test_val = distances[opti_coord[1], 1] - margin_height
    test_border = [test_val, opti_up]
    return test_border


#%%defining a new down border of the image
def best_border_coord_down(work_image, margin_height=30, half_matrix_size=2):
    # Define the best poistion to estimate the border (for the other coordinate)
    [N1, N2] = work_image.shape

    matrix_size = 2 * half_matrix_size + 1

    distances = np.zeros((N2 - half_matrix_size, 2), dtype=int)
    ones_matrix = np.ones((matrix_size, matrix_size))

    for i in range(half_matrix_size, N2 - half_matrix_size):
        done = False
        dist = N1 - half_matrix_size

        while done == False:

            around_dist = np.zeros((matrix_size, matrix_size), dtype=int)
            for n in range(i - half_matrix_size - 1, i + half_matrix_size):
                small_matrix_index_n = n % matrix_size
                for m in range(dist - half_matrix_size - 1, dist + half_matrix_size):
                    small_matrix_index_m = m % matrix_size
                    around_dist[
                        small_matrix_index_m, small_matrix_index_n
                    ] = work_image[m, n]

            if np.array_equal(around_dist, ones_matrix) or dist == half_matrix_size + 1:
                done = True
            else:
                dist = dist - 1
        distances[i - half_matrix_size] = dist

    distances = np.delete(distances, len(distances) - 1, 0)
    distances = np.delete(distances, len(distances) - 1, 0)

    opti_coord = np.argmax(distances, axis=0)
    opti_down = distances[opti_coord[1], 0]
    test_val = distances[opti_coord[1], 1] + margin_height
    test_border = [test_val, opti_down]
    return test_border


#%% defining a new left border of the image
def best_border_coord_left(work_image, margin_width=30, half_matrix_size=2):
    # Define the best poistion to estimate the border (for the other coordinate)
    [N1, N2] = work_image.shape

    matrix_size = 2 * half_matrix_size + 1

    distances = np.zeros((N1 - half_matrix_size, 2), dtype=int)
    ones_matrix = np.ones((matrix_size, matrix_size))

    for i in range(half_matrix_size, N1 - half_matrix_size):
        done = False
        dist = half_matrix_size
        distances[i - half_matrix_size, 0] = i
        while done == False:

            around_dist = np.zeros((matrix_size, matrix_size), dtype=int)
            for n in range(i - half_matrix_size, i + half_matrix_size + 1):
                small_matrix_index_n = n % matrix_size
                for m in range(dist - half_matrix_size, dist + half_matrix_size + 1):
                    small_matrix_index_m = m % matrix_size
                    around_dist[
                        small_matrix_index_n, small_matrix_index_m
                    ] = work_image[n, m]

            if np.array_equal(around_dist, ones_matrix) or dist == N2 - (
                half_matrix_size + 1
            ):
                done = True
            else:
                dist = dist + 1
        distances[i - half_matrix_size, 1] = dist

    distances = np.delete(distances, len(distances) - 1, 0)
    distances = np.delete(distances, len(distances) - 1, 0)

    opti_coord = np.argmin(distances, axis=0)
    opti_left = distances[opti_coord[1], 0]

    test_val = distances[opti_coord[1], 1] - margin_width
    test_border = [opti_left, test_val]
    return test_border


#%% defining a new right border of the image
def best_border_coord_right(work_image, margin_width=30, half_matrix_size=2):
    # Define the best poistion to estimate the border (for the other coordinate)
    [N1, N2] = work_image.shape

    matrix_size = 2 * half_matrix_size + 1

    distances = np.zeros((N1 - half_matrix_size, 2), dtype=int)
    ones_matrix = np.ones((matrix_size, matrix_size))

    for i in range(half_matrix_size, N1 - half_matrix_size):
        done = False
        dist = N2 - half_matrix_size
        distances[i - half_matrix_size, 0] = i
        while done == False:

            around_dist = np.zeros((matrix_size, matrix_size), dtype=int)
            for n in range(i - half_matrix_size - 1, i + half_matrix_size):
                small_matrix_index_n = n % matrix_size
                for m in range(dist - half_matrix_size - 1, dist + half_matrix_size):
                    small_matrix_index_m = m % matrix_size
                    around_dist[
                        small_matrix_index_n, small_matrix_index_m
                    ] = work_image[n, m]

            if np.array_equal(around_dist, ones_matrix) or dist == half_matrix_size + 1:
                done = True
            else:
                dist = dist - 1
        distances[i - half_matrix_size, 1] = dist

    distances = np.delete(distances, len(distances) - 1, 0)
    distances = np.delete(distances, len(distances) - 1, 0)

    opti_coord = np.argmax(distances, axis=0)
    opti_right = distances[opti_coord[1], 0]

    test_val = distances[opti_coord[1], 1] + margin_width
    test_border = [opti_right, test_val]
    return test_border


#%% cleaning the image. A point x is going through the image
def clean(work_image, half_matrix_size=2):

    [N1, N2] = work_image.shape
    clean_data = np.zeros((N1, N2), dtype=int)

    for i in range(N1):
        for j in range(N2):
            x = [i, j]
            diff_elem = []
            for m in range(i - half_matrix_size, i + half_matrix_size + 1):
                for n in range(j - half_matrix_size, j + half_matrix_size + 1):
                    if (n >= 0) and (n < N2) and (m >= 0) and (m < N1):
                        y = [m, n]
                        if y != x:
                            diff_elem.append(work_image[m, n])
            mode = sci.mode(diff_elem)
            clean_data[i, j] = mode[0]

    return clean_data


#%% finding the center of the mole (not of the image).
def mass_center(work_image):
    [N1, N2] = work_image.shape

    # Start with vertical matrices
    vertical_index = ma.floor(N2 / 2)
    old_vertical_index = N2
    vertical_center_found = False
    v_nb_iter = 0

    while vertical_center_found == False:

        left_matrix = work_image[0:N1, 0 : int(vertical_index)]
        right_matrix = work_image[0:N1, int(vertical_index) : N2]
        v_nb_iter = v_nb_iter + 1
        # print("Vertical index = " + str(vertical_index))

        # Count the mass of left matrix
        left_mass = 0
        [l_row, l_column] = left_matrix.shape
        for i in range(l_row):
            for j in range(l_column):
                left_mass = left_mass + left_matrix[i, j]
        # print("left mass = "+str(left_mass))

        # Count the mass of left matrix
        right_mass = 0
        [r_row, r_column] = right_matrix.shape
        for i in range(r_row):
            for j in range(r_column):
                right_mass = right_mass + right_matrix[i, j]
        # print("right mass = " + str(right_mass))

        if (right_mass == left_mass) or (v_nb_iter > 20):
            vertical_center_found = True

        elif right_mass > left_mass:
            new_vertical_index = vertical_index + (
                (abs(old_vertical_index - vertical_index)) / 2
            )
            old_vertical_index = vertical_index
            vertical_index = new_vertical_index

        elif right_mass < left_mass:
            new_vertical_index = vertical_index - (
                (abs(old_vertical_index - vertical_index)) / 2
            )
            old_vertical_index = vertical_index
            vertical_index = new_vertical_index

    # Same method apply for horizontal matrices
    horizontal_index = ma.floor(N1 / 2)
    old_horizontal_index = N1
    horizontal_center_found = False
    h_nb_iter = 0

    while horizontal_center_found == False:

        up_matrix = work_image[0 : int(horizontal_index), 0:N2]
        down_matrix = work_image[int(horizontal_index) : N1, 0:N2]
        h_nb_iter = h_nb_iter + 1
        # print("Horizontal index = "+str(horizontal_index))

        # Count mass of up matrix
        up_mass = 0
        [u_row, u_column] = up_matrix.shape
        for i in range(u_row):
            for j in range(u_column):
                up_mass = up_mass + up_matrix[i, j]
        # print("up mass = " + str(up_mass))

        # Count mass of up matrix
        down_mass = 0
        [d_row, d_column] = down_matrix.shape
        for i in range(d_row):
            for j in range(d_column):
                down_mass = down_mass + down_matrix[i, j]
        # print("down mass = " + str(down_mass))

        if (up_mass == down_mass) or (h_nb_iter > 20):
            horizontal_center_found = True

        elif up_mass < down_mass:
            new_horizontal_index = horizontal_index + (
                (abs(old_horizontal_index - horizontal_index)) / 2
            )
            old_horizontal_index = horizontal_index
            horizontal_index = new_horizontal_index

        elif up_mass > down_mass:
            new_horizontal_index = horizontal_index - (
                (abs(old_horizontal_index - horizontal_index)) / 2
            )
            old_horizontal_index = horizontal_index
            horizontal_index = new_horizontal_index

    center = [int(horizontal_index), int(vertical_index)]
    # work_image[:,int(vertical_index)] = 4
    # work_image[int(horizontal_index),:] = 4
    # show_image(work_image, "mass center of the mole")
    return center


#%% verifying if the mole border is continuous or not
def connexity(work_image, half_matrix_size=3):
    [N1, N2] = work_image.shape

    mid_row = ma.floor(N1 / 2)
    mid_col = ma.floor(N2 / 2)

    matrix_size = 2 * half_matrix_size + 1

    connexity = True

    row_found = False
    col_found = False
    y = ma.floor(N2 / 2) + 5
    while col_found == False:
        i = 1
        while row_found == False:
            if work_image[i, y] == 2:
                start_row = i
                row_found = True
            else:
                i = i + 1
        if work_image[i, y] == 2:
            start_col = y
            col_found = True
        else:
            y = y + 1

    start_point = [start_row, start_col]
    start_point_reached = False
    current_point = start_point
    studied_point = [current_point]
    it = 0

    while (start_point_reached == False) and (connexity == True):

        start_point_inside_current_matrix = False
        if it > 10:
            for i in range(
                current_point[0] - half_matrix_size,
                current_point[0] + half_matrix_size + 1,
            ):
                for j in range(
                    current_point[1] - half_matrix_size,
                    current_point[1] + half_matrix_size + 1,
                ):
                    if (i == start_point[0]) and (j == start_point[1]):
                        start_point_inside_current_matrix = True
                        print("\nfound !")

        if start_point_inside_current_matrix == True:
            start_point_reached = True
        else:

            around_current_point = np.zeros((matrix_size, matrix_size), dtype=int)
            small_matrix_i = 0
            for i in range(
                current_point[0] - half_matrix_size,
                current_point[0] + half_matrix_size + 1,
            ):
                small_matrix_j = 0
                for j in range(
                    current_point[1] - half_matrix_size,
                    current_point[1] + half_matrix_size + 1,
                ):
                    around_current_point[small_matrix_i, small_matrix_j] = work_image[
                        i, j
                    ]
                    small_matrix_j = small_matrix_j + 1
                small_matrix_i = small_matrix_i + 1

            # print(around_current_point)

            mass_of_border = 0
            for i in range(matrix_size):
                for j in range(matrix_size):
                    if around_current_point[i, j] == 2:
                        mass_of_border = mass_of_border + 1

            if mass_of_border >= 0:

                col_to_add = 0
                row_to_add = 0
                if (current_point[0] <= mid_row) and (current_point[1] > mid_col):
                    # print("on est dans le carré 1\n")
                    for i in range(matrix_size):
                        if around_current_point[i, matrix_size - 1] == 2:
                            row_to_add = i - half_matrix_size
                            col_to_add = matrix_size - 1 - half_matrix_size
                        elif (row_to_add == 0) & (col_to_add == 0):
                            for j in range(matrix_size):
                                if around_current_point[matrix_size - 1, j] == 2:
                                    row_to_add = matrix_size - 1 - half_matrix_size
                                    col_to_add = j - half_matrix_size
                                elif (row_to_add == 0) & (col_to_add == 0):
                                    for k in range(matrix_size):
                                        if around_current_point[k, 0] == 2:
                                            row_to_add = k - half_matrix_size
                                            col_to_add = 0 - half_matrix_size
                                        elif (row_to_add == 0) & (col_to_add == 0):
                                            for l in range(matrix_size):
                                                if around_current_point[0, l] == 2:
                                                    row_to_add = 0 - half_matrix_size
                                                    col_to_add = l - half_matrix_size
                    next_point = [
                        current_point[0] + row_to_add,
                        current_point[1] + col_to_add,
                    ]

                elif (current_point[0] > mid_row) and (current_point[1] >= mid_col):
                    # print("on est dans le carré 2\n")
                    for i in range(matrix_size):
                        if around_current_point[matrix_size - 1, i] == 2:
                            row_to_add = matrix_size - 1 - half_matrix_size
                            col_to_add = i - half_matrix_size
                        elif (row_to_add == 0) & (col_to_add == 0):
                            for j in range(matrix_size):
                                if around_current_point[j, 0] == 2:
                                    row_to_add = j - half_matrix_size
                                    col_to_add = 0 - half_matrix_size
                                elif (row_to_add == 0) & (col_to_add == 0):
                                    for k in range(matrix_size):
                                        if around_current_point[0, k] == 2:
                                            row_to_add = 0 - half_matrix_size
                                            col_to_add = k - half_matrix_size
                                        elif (row_to_add == 0) & (col_to_add == 0):
                                            for l in range(matrix_size):
                                                if (
                                                    around_current_point[
                                                        l, matrix_size - 1
                                                    ]
                                                    == 2
                                                ):
                                                    row_to_add = l - half_matrix_size
                                                    col_to_add = (
                                                        matrix_size
                                                        - 1
                                                        - half_matrix_size
                                                    )
                    next_point = [
                        current_point[0] + row_to_add,
                        current_point[1] + col_to_add,
                    ]

                elif (current_point[0] >= mid_row) and (current_point[1] < mid_col):
                    # print("on est dans le carré 3\n")
                    for i in range(matrix_size):
                        if around_current_point[i, 0] == 2:
                            row_to_add = i - half_matrix_size
                            col_to_add = 0 - half_matrix_size
                        elif (row_to_add == 0) & (col_to_add == 0):
                            for j in range(matrix_size):
                                if around_current_point[0, j] == 2:
                                    row_to_add = 0 - half_matrix_size
                                    col_to_add = j - half_matrix_size
                                elif (row_to_add == 0) & (col_to_add == 0):
                                    for k in range(matrix_size):
                                        if (
                                            around_current_point[k, matrix_size - 1]
                                            == 2
                                        ):
                                            row_to_add = k - half_matrix_size
                                            col_to_add = (
                                                matrix_size - 1 - half_matrix_size
                                            )
                                        elif (row_to_add == 0) & (col_to_add == 0):
                                            for l in range(matrix_size):
                                                if (
                                                    around_current_point[
                                                        matrix_size - 1, l
                                                    ]
                                                    == 2
                                                ):
                                                    row_to_add = (
                                                        matrix_size
                                                        - 1
                                                        - half_matrix_size
                                                    )
                                                    col_to_add = l - half_matrix_size
                    next_point = [
                        current_point[0] + row_to_add,
                        current_point[1] + col_to_add,
                    ]

                elif (current_point[0] < mid_row) and (current_point[1] <= mid_col):
                    # print("on est dans le carré 4\n")
                    for i in range(matrix_size):
                        if around_current_point[0, i] == 2:
                            row_to_add = 0 - half_matrix_size
                            col_to_add = i - half_matrix_size
                        elif (row_to_add == 0) & (col_to_add == 0):
                            for j in range(matrix_size):
                                if around_current_point[j, matrix_size - 1] == 2:
                                    row_to_add = j - half_matrix_size
                                    col_to_add = matrix_size - 1 - half_matrix_size
                                elif (row_to_add == 0) & (col_to_add == 0):
                                    for k in range(matrix_size):
                                        if (
                                            around_current_point[matrix_size - 1, k]
                                            == 2
                                        ):
                                            row_to_add = (
                                                matrix_size - 1 - half_matrix_size
                                            )
                                            col_to_add = k - half_matrix_size
                                        elif (row_to_add == 0) & (col_to_add == 0):
                                            for l in range(matrix_size):
                                                if around_current_point[l, 0] == 2:
                                                    row_to_add = l - half_matrix_size
                                                    col_to_add = 0 - half_matrix_size
                    next_point = [
                        current_point[0] + row_to_add,
                        current_point[1] + col_to_add,
                    ]

                else:
                    print("y a une merde")
                    break

                work_image[current_point[0], current_point[1]] = 3

                # print(studied_point)
                # print(current_point)
                for i in studied_point:
                    if i == next_point:
                        print("on recule !")
                        connexity = False

                current_point = next_point
                studied_point.append(current_point)
                it = it + 1
            else:
                connexity = False

    if connexity:
        print("\nThe mole has continuous border\n")
        return True
    else:
        print("\nThe mole hasn't continuous boreder\n")
        return False


#%% defining the perimeter and the area of the mole


def border(work_image):
    [N1, N2] = work_image.shape

    # up border
    for i in range(N2):
        up_done = False
        occurs = 1
        while up_done == False:
            if (work_image[occurs, i] == 1) or (occurs == N1 - 1):
                up_done = True
                if occurs != N1 - 1:
                    work_image[occurs - 1, i] = 2
                # if ((work_image[occurs - 1, i] == 0) and (work_image[occurs, i] == 1) and (work_image[occurs + 1, i] == 1)):
            else:
                occurs = occurs + 1

    # down border
    for i in range(N2):
        down_done = False
        occurs = N1 - 2
        while down_done == False:
            if (work_image[occurs, i] == 1) or (occurs == 1):
                down_done = True
                if occurs != 1:
                    work_image[occurs + 1, i] = 2
                # if ((work_image[occurs - 1, i] == 1) and (work_image[occurs, i] == 1) and (work_image[occurs + 1, i] == 0)):
            else:
                occurs = occurs - 1

    # left border
    for i in range(N1):
        left_done = False
        occurs = 1
        while left_done == False:
            if (work_image[i, occurs] == 1) or (occurs == N2 - 1):
                left_done = True
                if occurs != N2 - 1:
                    work_image[i, occurs - 1] = 2
                # if ((work_image[occurs - 1, i] == 0) and (work_image[occurs, i] == 1) and (work_image[occurs + 1, i] == 1)):
            else:
                occurs = occurs + 1

    # right border
    for i in range(N1):
        right_done = False
        occurs = N2 - 2
        while right_done == False:
            if (work_image[i, occurs] == 1) or (occurs == 1):
                right_done = True
                if occurs != 1:
                    work_image[i, occurs + 1] = 2
                # if ((work_image[i, occurs - 1] == 1) and (work_image[i, occurs] == 1) and (work_image[i, occurs + 1] == 0)):
            else:
                occurs = occurs - 1

    show_image(work_image, "bordered image")

    # Count every point inside the defined perimeter to define the area of the mole
    mole_area = 0
    mole_perimeter = 0
    for i in range(N1):
        for j in range(N2):
            if work_image[i, j] == 1:
                mole_area = mole_area + 1
            if work_image[i, j] == 2:
                mole_perimeter = mole_perimeter + 1

    # area = pi * R^2 and perimeter = 2 * pi * R
    ideal_circle_ray = ma.sqrt(mole_area / ma.pi)
    ideal_circle_perimeter = 2 * ma.pi * ideal_circle_ray

    perimeter_ratio = mole_perimeter / ideal_circle_perimeter
    #    print("\nThe perimeter of the mole is : ",mole_perimeter)
    #    print("The perimeter of a perfect circle with the same area as the mole is : ",ideal_circle_perimeter)
    #    print("\nThe obtained ratio between the perimeter of the mole and the perimeter of the corresponding circle is:\n"
    #          "The higher is this value more dangerous is the mole\n"
    #          "The usual values of this ratio are around 1\n\n",perimeter_ratio)
    return perimeter_ratio


#%% This function aims to verifying if the mole is symmetrical or not
def symmetrie(work_image, mass_center):

    [N1, N2] = work_image.shape

    vertical_index = mass_center[1]
    horizontal_index = mass_center[0]
    #   _________
    matrix_1 = work_image[0:horizontal_index, vertical_index:N2]  #  |    |    |
    matrix_2 = work_image[horizontal_index:N1, vertical_index:N2]  #  |_4__|__1_|
    matrix_3 = work_image[horizontal_index:N1, 0:vertical_index]  #  | 3  |  2 |
    matrix_4 = work_image[0:horizontal_index, 0:vertical_index]  #  |____|____|

    # Comparison of matrix 4 and matrix 2

    reverse_matrix_2 = np.fliplr(np.flipud(matrix_2))

    # show_image(matrix_4, "upper left")
    # show_image(reverse_matrix_2, "lower right")

    if matrix_4.shape != reverse_matrix_2.shape:
        if matrix_4.shape[0] < reverse_matrix_2.shape[0]:
            if matrix_4.shape[1] < reverse_matrix_2.shape[1]:
                # Cas 1
                row_to_add = reverse_matrix_2.shape[0] - matrix_4.shape[0]
                col_to_add = reverse_matrix_2.shape[1] - matrix_4.shape[1]

                new_matrix_4 = np.pad(
                    matrix_4,
                    ((row_to_add, 0), (col_to_add, 0)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                sub_matrix = np.subtract(new_matrix_4, reverse_matrix_2)
            else:
                # Cas 2
                row_to_add = reverse_matrix_2.shape[0] - matrix_4.shape[0]
                col_to_add = matrix_4.shape[1] - reverse_matrix_2.shape[1]

                new_matrix_4 = np.pad(
                    matrix_4,
                    ((row_to_add, 0), (0, 0)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                new_reverse_matrix_2 = np.pad(
                    reverse_matrix_2,
                    ((0, 0), (col_to_add, 0)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                sub_matrix = np.subtract(new_matrix_4, new_reverse_matrix_2)
        else:
            if matrix_4.shape[1] < reverse_matrix_2.shape[1]:
                # Cas 3
                row_to_add = matrix_4.shape[0] - reverse_matrix_2.shape[0]
                col_to_add = reverse_matrix_2.shape[1] - matrix_4.shape[1]

                new_matrix_4 = np.pad(
                    matrix_4,
                    ((0, 0), (col_to_add, 0)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                new_reverse_matrix_2 = np.pad(
                    reverse_matrix_2,
                    ((row_to_add, 0), (0, 0)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                sub_matrix = np.subtract(new_matrix_4, new_reverse_matrix_2)

            else:
                # Cas 4
                row_to_add = matrix_4.shape[0] - reverse_matrix_2.shape[0]
                col_to_add = matrix_4.shape[1] - reverse_matrix_2.shape[1]

                new_reverse_matrix_2 = np.pad(
                    reverse_matrix_2,
                    ((row_to_add, 0), (col_to_add, 0)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                sub_matrix = np.subtract(matrix_4, new_reverse_matrix_2)
    else:
        sub_matrix = np.subtract(matrix_4, reverse_matrix_2)

    [row, col] = sub_matrix.shape

    remain_mass = 0
    for i in range(row):
        for j in range(col):
            remain_mass = remain_mass + abs(sub_matrix[i, j])

    percentage = abs(remain_mass / np.size(sub_matrix))
    #    print("\nThe asymmetry percentage between matrices 2 and 4 is : " + str(percentage*100) + "% (lower is better)")
    if percentage < 0.05:
        symmetry_4_2 = True
    else:
        symmetry_4_2 = False

    # Comparison of matrix 1 and matrix 3

    reverse_matrix_3 = np.fliplr(np.flipud(matrix_3))

    # show_image(matrix_1, "upper right")
    # show_image(reverse_matrix_3, "lower left")

    if matrix_1.shape != reverse_matrix_3.shape:
        if matrix_1.shape[0] < reverse_matrix_3.shape[0]:
            if matrix_1.shape[1] < reverse_matrix_3.shape[1]:
                # Cas 1
                row_to_add1 = reverse_matrix_3.shape[0] - matrix_1.shape[0]
                col_to_add1 = reverse_matrix_3.shape[1] - matrix_1.shape[1]

                new_matrix_1 = np.pad(
                    matrix_1,
                    ((row_to_add1, 0), (0, col_to_add1)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                sub_matrix2 = np.subtract(new_matrix_1, reverse_matrix_3)
            else:
                # Cas 2
                row_to_add1 = reverse_matrix_3.shape[0] - matrix_1.shape[0]
                col_to_add1 = matrix_1.shape[1] - reverse_matrix_3.shape[1]

                new_matrix_1 = np.pad(
                    matrix_1,
                    ((row_to_add1, 0), (0, 0)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                new_reverse_matrix_3 = np.pad(
                    reverse_matrix_3,
                    ((0, 0), (0, col_to_add1)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                sub_matrix2 = np.subtract(new_matrix_1, new_reverse_matrix_3)
        else:
            if matrix_1.shape[1] < reverse_matrix_3.shape[1]:
                # Cas 3
                row_to_add1 = matrix_1.shape[0] - reverse_matrix_3.shape[0]
                col_to_add1 = reverse_matrix_3.shape[1] - matrix_1.shape[1]

                new_matrix_1 = np.pad(
                    matrix_1,
                    ((0, 0), (0, col_to_add1)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                new_reverse_matrix_3 = np.pad(
                    reverse_matrix_3,
                    ((row_to_add1, 0), (0, 0)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                sub_matrix2 = np.subtract(new_matrix_1, new_reverse_matrix_3)

            else:
                # Cas 4
                row_to_add1 = matrix_1.shape[0] - reverse_matrix_3.shape[0]
                col_to_add1 = matrix_1.shape[1] - reverse_matrix_3.shape[1]

                new_reverse_matrix_3 = np.pad(
                    reverse_matrix_3,
                    ((row_to_add1, 0), (0, col_to_add1)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                sub_matrix2 = np.subtract(matrix_1, new_reverse_matrix_3)
    else:
        sub_matrix2 = np.subtract(matrix_1, reverse_matrix_3)

    [row2, col2] = sub_matrix2.shape

    remain_mass2 = 0
    for i in range(row2):
        for j in range(col2):
            remain_mass2 = remain_mass2 + abs(sub_matrix2[i, j])

    percentage2 = abs(remain_mass2 / np.size(sub_matrix2))
    #    print("\nThe asymmetry percentage between matrices 3 and 1 is : " + str(percentage2*100) + "% (lower is better)")
    if percentage2 < 0.05:
        symmetry_3_1 = True
    else:
        symmetry_3_1 = False

    global_percentage = (percentage + percentage2) / 2
    #    print("\nThe mean of the two obtained percentages is : " + str(global_percentage*100) + "% (lower is better)")
    if global_percentage < 0.05:
        symmetry_glob = True
    else:
        symmetry_glob = False

    if (symmetry_4_2 and symmetry_3_1) or symmetry_glob:
        print("\nThe mole is globally symmetrical (less than 5%)")
    else:
        print("\nThe mole is not globally symmetrical (more than 5%)")

    return global_percentage


if __name__ == "__main__":

    images = load_images()
    image = mpimg.imread("data/medium_risk_3_s.jpg")
    glob_res = []
    res = [3, 0, 0, 0, 0, 0]
    res2 = [3, 1.9, 1, 0.043, 1, 0]

    # show_image(image, "Original image")

    work_image = quantize_picture(image, nb_of_clusters=4)
    show_image(work_image, "Quantized image")

    new_work_image = resize_picture(work_image, margin_height=15, margin_width=15)
    show_image(new_work_image, "Resized image")

    new_work_image2 = clean(new_work_image, half_matrix_size=3)
    show_image(new_work_image2, "Clean quantized image")
    new_work_image3 = cp.deepcopy(new_work_image2)

    ratio = border(new_work_image2)
    res[1] = ratio
    con = connexity(new_work_image2, half_matrix_size=4)
    if con == True:
        res[2] = 1
    else:
        res[2] = 0
    show_image(new_work_image2, "Center of mass")

    center = mass_center(new_work_image3)
    print("\nthe mass center is : ", center)

    sym = symmetrie(new_work_image3, center)
    res[3] = sym * 100
    if sym < 0.05:
        res[4] = 1
    else:
        res[4] = 0

    res[5] = 2 - (res[2] + res[4])

    # glob_res.append(res)
    # glob_res.append(res2)
    # np.savetxt("res.txt", glob_res)
