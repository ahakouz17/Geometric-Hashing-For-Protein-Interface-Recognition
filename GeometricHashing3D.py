# This code handles 3D model-based objects recognition using geometric hashing algorithm
# Authors: Asma Hakouz, Dalyah jamal, Dzenaida gicic
# 11/1/2018


import math
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
from random import shuffle


class TransformationParams:
    def __init__(self, angle_x, angle_y, angle_z, scale, offset_point):
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.scale_factor = scale
        self.offset = offset_point


class Point3D:
    def __init__(self, x, y, z, pid):
        self.x = x
        self.y = y
        self.z = z
        self.id = pid


def calculate_transformation_parameters(point1, point2, point3, bin_size):
    # this function calculates the transformation parameters for the bases (point1, point2 and point3)
    # it returns a TransformationParams object which contains info about rotation angles, translation offset and scale
    # factor
    x1, y1, z1 = float(point1.x), float(point1.y), float(point1.z)
    x2, y2, z2 = float(point2.x), float(point2.y), float(point2.z)
    x3, y3, z3 = float(point3.x), float(point3.y), float(point3.z)

    # ### TRANSLATION ####
    # calculate offset/translation and apply it to the bases
    new_x2, new_y2, new_z2 = x2 - x1, y2 - y1, z2 - z1
    new_x3, new_y3, new_z3 = x3 - x1, y3 - y1, z3 - z1

    # check that the chosen bases p1, p2, and p3 are not collinear by checking that vector R12 is not a scaled version
    # of R13, assuming that p1 is the origin (0,0,0)
    if new_x2 * new_z3 == new_x3 * new_z2 and new_x2 * new_y3 == new_x3 * new_y2 and new_y2 * new_z3 == new_y3 * new_z2:
        # print("Points are not collinear")
        return None

    # ### ROTATION ###
    # now since our first point is centered at the origin of our new coordinate system, we want to transform others so
    # that the second point will land on the positive x-axis at point 1,0,0 (after normalizing by the scale factor)
    # while the third point will land on the xy-plane. To do so we will first perform rotation around the y-axis so
    # that the 2nd point lands on the xy plane. Then, we'll perform rotation around z-axis so that point 2 will land on
    # positive x-axis. Finally, we'll do rotation around x-axis so that the third point lands on the xy plane

    # first step is to calculate the rotation angle around the y-axis
    if new_x2 == 0:
        if new_z2 == 0:
            rot_y_rad = 0 # point 2 is already on the xy plane
        elif new_z2 > 0:
            rot_y_rad = math.pi / 2.0
        else:
            rot_y_rad = -1 * math.pi / 2.0
    else:
        rot_y_rad = math.atan2(new_z2, new_x2) # rotation angle around y-axis in radians


    # Apply rotation around y-axis for the points
    if rot_y_rad != 0:
        sin_theta = math.sin(rot_y_rad)
        cos_theta = math.cos(rot_y_rad)

        # since we're rotating around y-axis, y coordinates won't be affected
        new_x2 = new_x2 * cos_theta + new_z2 * sin_theta
        new_z2 = -1 * new_x2 * sin_theta + new_z2 * cos_theta  # this should be zero

        new_x3 = new_x3 * cos_theta + new_z3 * sin_theta
        new_z3 = -1 * new_x3 * sin_theta + new_z3 * cos_theta

    # now, rotate around z-axis so that point 2 will land on the positive x-axis
    if new_x2 == 0:
        if new_y2 == 0:
            rot_z_rad = 0  # point 2 is already on the positive x-axis
        elif new_y2 > 0:
            rot_z_rad = math.pi / 2.0
        else:
            rot_z_rad = -1 * math.pi / 2.0
    else:
        rot_z_rad = math.atan2(new_y2, new_x2)

    if rot_z_rad != 0:
        sin_theta = math.sin(-1 * rot_z_rad)
        cos_theta = math.cos(-1 * rot_z_rad)
        new_x2 = new_x2 * cos_theta + -1 * new_y2 * sin_theta
        new_y2 = new_x2 * sin_theta + new_y2 * cos_theta

        new_x3 = new_x3 * cos_theta + -1 * new_y3 * sin_theta
        new_y3 = new_x3 * sin_theta + new_y3 * cos_theta

    # finally, calculate rotation around x-axis, such that the third point lands on the xy plane
    if new_y3 == 0:
        if new_z3 == 0:
            rot_x_rad = 0   # point 3 is already on the xy plane
        elif new_z3 > 0:
            rot_x_rad = math.pi / 2.0  # 90 degrees
        else:
            rot_x_rad = -1 * math.pi / 2.0  # -90 degrees
    else:
        rot_x_rad = math.atan2(new_z3, new_y3)

    scale = new_x2

    # we need to scale our new reference coordinate system so that the distance between the first two points (bases)
    # is 1
    trans_info = TransformationParams(rot_x_rad, rot_y_rad, rot_z_rad, scale, point1)

    return trans_info


def transform_point3d(trans_info, p, bin_size):
    # this function transforms a point from the original to the new coordinate system using the
    # transformation details provided in transformation_info object

    # Apply translation
    p.x = p.x - trans_info.offset.x
    p.y = p.y - trans_info.offset.y
    p.z = p.z - trans_info.offset.z

    # Rotate around y-axis
    sin_theta = math.sin(trans_info.angle_y)
    cos_theta = math.cos(trans_info.angle_y)
    p.x = p.x * cos_theta + p.z * sin_theta
    p.z = -1 * p.x * sin_theta + p.z * cos_theta

    # Rotate around z-axis
    sin_theta = math.sin(-1 * trans_info.angle_z)
    cos_theta = math.cos(-1 * trans_info.angle_z)
    p.x = p.x * cos_theta + -1 * p.y * sin_theta
    p.z = p.x * sin_theta + p.y * cos_theta

    # Rotate around x-axis
    sin_theta = math.sin(-1 * trans_info.angle_x)
    cos_theta = math.cos(-1 * trans_info.angle_x)
    p.y = p.y * cos_theta + -1 * p.z * sin_theta
    p.z = p.y * sin_theta + p.z * cos_theta

    p.x = int(round(p.x / bin_size)) / trans_info.scale_factor
    p.y = int(round(p.y / bin_size)) / trans_info.scale_factor
    p.z = int(round(p.z / bin_size)) / trans_info.scale_factor

    return p


def add_entry_to_hash_table(htable, point, b1, b2, b3, model_id, include_id):
    # this function adds an entry to link the x, y coordinates of the point to the model ID and bases pairs info
    if type(htable) != dict:
        # this should not happen, added just in case
        htable = {}

    # check if there is already an entry in the dictionary for the point's x coordinate
    # if not create a new empty entry
    if point.x not in htable:
        htable[point.x] = {}

    # check if there is already an entry for the point's y coordinate inside point.x's entry if not
    # create a new empty entry
    if point.y not in htable[point.x]:
        htable[point.x][point.y] = {}

    if point.z not in htable[point.x][point.y]:
        htable[point.x][point.y][point.z] = {}

    # check if there is already an entry for the model coordinate inside the point.x, point.y entry
    # if not create a new empty entry
    if model_id not in htable[point.x][point.y][point.z]:
        htable[point.x][point.y][point.z][model_id] = []

    # add the entry in the x,y bin with model ID and bases pair
    if include_id:
        htable[point.x][point.y][point.z][model_id].append([b1, b2, b3, point.id])
    else:
        htable[point.x][point.y][point.z][model_id].append([b1, b2, b3])

    return htable


def compare_for_clustering(cluster_rep_table, new_model_table, model_lengths, thresh):
    # this function is used to compare the hash table of a new model to the hash table of
    # a the model that is the cluster representative to check for matching.
    votes_table = {}

    # vote for x, y coordinates from  new_model_table that also exist in cluster_rep_table
    for x in new_model_table:
        if x not in cluster_rep_table:
            return False

        for y in new_model_table[x]:
            if y not in cluster_rep_table[x]:
                return False

            for z in new_model_table[x][y]:
                if z not in cluster_rep_table[x][y]:
                    return False

                for model_id in cluster_rep_table[x][y][z]:
                    for b1, b2, b3 in cluster_rep_table[x][y][z][model_id]:
                        if model_id not in votes_table:
                            votes_table[model_id] = {}

                        if b1 not in votes_table[model_id]:
                            votes_table[model_id][b1] = {}

                        if b2 not in votes_table[model_id][b1]:
                            votes_table[model_id][b1][b2] = {}

                        if b3 not in votes_table[model_id][b1][b2]:
                            votes_table[model_id][b1][b2][b3] = 0

                        votes_table[model_id][b1][b2][b3] += (1.0 / model_lengths[model_id])

    max_vote = 0
    for model_id in votes_table:
        for b1 in votes_table[model_id].keys():
            for b2 in votes_table[model_id][b1]:
                for b3 in votes_table[model_id][b1][b2]:
                    if votes_table[model_id][b1][b2][b3] > max_vote:
                        max_vote = votes_table[model_id][b1][b2][b3]

    # check if a match exists. i.e., if the number of max votes is above the threshold
    if max_vote >= thresh:
        return True
    else:
        return False


def merge_with_full_htable(full_htable, cluster_htable):
    for x_sub in cluster_htable:
        for y_sub in cluster_htable[x_sub]:
            for z_sub in cluster_htable[x_sub][y_sub]:
                model = cluster_htable[x_sub][y_sub][z_sub].keys()[0]
                if x_sub not in full_htable:
                    full_htable[x_sub] = {}
                if y_sub not in full_htable[x_sub]:
                    full_htable[x_sub][y_sub] = {}
                if z_sub not in full_htable[x_sub][y_sub]:
                    full_htable[x_sub][y_sub][z_sub] = {}
                full_htable[x_sub][y_sub][z_sub][model] = cluster_htable[x_sub][y_sub][z_sub][model]
    return full_htable


def start_offline_training(training_threshold, bin_size):
    # this function is the offline training part where we loop through the models and produce the
    # full hash table. Also, a clustering step was added to eliminate similar models by clustering
    # them together and choosing only one model to be the cluster representative then add its hash
    # table to the full table
    print('Offline learning started...')
    beginning = datetime.now()
    print(beginning)

    filenames = []
    for root, dirs, files in os.walk("models3D/models/"):
        for filename in files:
            filenames.append(filename)

    model_lengths = [0] * len(filenames)

    # initialization
    num_unclustered_models = len(filenames)
    cluster_ids = {}
    model_id = -1
    full_htable = {}
    num_of_models = len(filenames)

    # we look for an unclustered Model, pick it as the cluster representative, construct its hash table, add it to
    # the full table. then find all matches of this cluster and eliminate them
    for b1 in range(0, num_of_models):
        cluster_ids[filenames[b1]] = -1  # 0 means that this fragment is not clustered yet

    while num_unclustered_models > 0:
        model_id += 1 # cluster id
        for k in range(0, num_of_models):
            if cluster_ids[filenames[k]] == -1:  # find the first unclustered model
                cluster_ids[filenames[k]] = model_id
                cluster_htable = {}
                break

        model_file = open('models3D/models/' + filenames[k], 'r')
        lines = model_file.readlines()
        model_file.close()

        p_coord = [[], [], [], [], []]
        #Feature Extraction
        for ii in range(0, len(lines)):
            # fetch atom data record
            record = lines[ii].split()
            # Only read alpha carbon atoms
            if not lines[ii].startswith("ATOM"):
                continue
            if record[2] != "CA":
                continue
            # extract data from ATOM record
            p_coord[0].append(float(record[6]))
            p_coord[1].append(float(record[7]))
            p_coord[2].append(float(record[8]))
            p_coord[3].append(-1)  # current identified model
            p_coord[4].append(-1)  # identified model ID

        model_lengths[k] = len(p_coord[0]) - 1

        # creates the full geometric hash table of the model
        for b1 in range(0, len(p_coord[0])):
            print(b1)
            x1, y1, z1 = p_coord[0][b1], p_coord[1][b1], p_coord[2][b1]
            p1 = Point3D(float(x1), float(y1), float(z1), b1)

            for b2 in range(0, len(p_coord[0])):
                if b1 != b2:
                    x2, y2, z2 = p_coord[0][b2], p_coord[1][b2], p_coord[2][b2]
                    p2 = Point3D(float(x2), float(y2), float(z2), b2)

                    for b3 in range(0, len(p_coord[0])):
                        if b1 != b3 and b2 != b3:
                            x3, y3, z3 = p_coord[0][b3], p_coord[1][b3], p_coord[2][b3]
                            p3 = Point3D(float(x3), float(y3), float(z2), b3)

                            # check that we're picking two unique points as bases
                            trans_info = calculate_transformation_parameters(p1, p2, p3, bin_size)

                            if trans_info is None:
                                # points are collinear, pick another combination
                                continue
                            # after picking the bases, loop through the remaining points
                            for p_index in range(0, len(p_coord[0])):
                                if p_index != b1:
                                    point_x, point_y, point_z = p_coord[0][p_index], p_coord[1][p_index], p_coord[2][p_index]
                                    point = Point3D(float(point_x), float(point_y), float(point_z), p_index)
                                    new_point = transform_point3d(trans_info, point, bin_size) # transformation and quantization
                                    cluster_htable = add_entry_to_hash_table(cluster_htable, new_point, b1, b2, b3, model_id, False)

        num_unclustered_models -= 1
        merge_with_full_htable(full_htable, cluster_htable)

        # cluster all matching models
        for n in range(k + 1, num_of_models):  # loop through the remaining models
            model_lengths[n] = len(p_coord[0]) - 1
            if cluster_ids[filenames[n]] == -1:
                model_file = open('models3D/models/' + filenames[n], 'r')
                lines = model_file.readlines()
                model_file.close()
                matched = False

                for b1 in range(0, len(p_coord[0])):
                    x1, y1, z1 = p_coord[0][b1], p_coord[1][b1], p_coord[2][b1]
                    p1 = Point3D(float(x1), float(y1), float(z1), b1)

                    for b2 in range(0, len(p_coord[0])):
                        if b1 != b2:
                            x2, y2, z2 = p_coord[0][b2], p_coord[1][b2], p_coord[2][b2]
                            p2 = Point3D(float(x2), float(y2), float(z2), b2)

                            for b3 in range(0, len(p_coord[0])):
                                if b1 != b3 and b2 != b3:
                                    x3, y3, z3 = p_coord[0][b3], p_coord[1][b3], p_coord[2][b3]
                                    p3 = Point3D(float(x3), float(y3), float(z2), b3)

                                trans_info = calculate_transformation_parameters(p1, p2, p3, bin_size)
                                if trans_info is None:
                                    # points are collinear, pick another combination
                                    continue

                                model_htabel = {}

                                for p_index in range(0, len(p_coord[0])):# after picking the bases, loop through remaining points
                                    if p_index != b1:
                                        point_x, point_y, point_z = p_coord[0][p_index], p_coord[1][p_index], p_coord[2][p_index]
                                        point = Point3D(float(point_x), float(point_y), float(point_z), p_index)
                                        new_point = transform_point3d(trans_info, point, bin_size)
                                        model_htabel = add_entry_to_hash_table(model_htabel, new_point, b1, b2, b3, model_id, False)
                                matched = compare_for_clustering(cluster_htable, model_htabel, model_lengths, training_threshold)
                                # print(matched)
                                if matched:  # if the base fragment and the new one are similar
                                    cluster_ids[filenames[n]] = model_id
                                    num_unclustered_models -= 1
                                    break
                        if matched:
                            break
                    if matched:
                        break

    print('Clustering ended...')
    end = datetime.now()
    print(end)
    duration = end - beginning
    hours, mod = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(mod, 60)
    print('Duration: ' + str(hours) + 'h, ' + str(minutes) + 'm, ' + str(seconds) + 's')
    return full_htable, model_lengths, filenames


def search_for_a_match(full_htable, test_htable, threshold, model_lengths):
    # this function is used to compare the hash table of a new model to the hash table of
    # a the model that is the cluster representative to check for matching.
    votes_count = {}
    votes_lists = {}

    # vote for x, y coordinates from  test_htable that also exist in full_htable
    for x in test_htable:
        if x not in full_htable:
            continue

        for y in test_htable[x]:
            if y not in full_htable[x]:
                continue

            for z in test_htable[x][y]:
                if z not in full_htable[x][y]:
                    continue

                for model_id in full_htable[x][y][z]:
                    for b1, b2, b3 in full_htable[x][y][z][model_id]:
                        if model_id not in votes_count:
                            votes_count[model_id] = {}
                            votes_lists[model_id] = {}

                        if b1 not in votes_count[model_id]:
                            votes_count[model_id][b1] = {}
                            votes_lists[model_id][b1] = {}

                        if b2 not in votes_count[model_id][b1]:
                            votes_count[model_id][b1][b2] = {}
                            votes_lists[model_id][b1][b2] = {}

                        if b3 not in votes_count[model_id][b1][b2]:
                            votes_count[model_id][b1][b2][b3] = 0
                            votes_lists[model_id][b1][b2][b3] = []
                        if test_htable[x][y][z][0][0][3] not in votes_lists[model_id][b1][b2][b3]:
                            votes_count[model_id][b1][b2][b3] += (1.0 / model_lengths[model_id])
                            votes_lists[model_id][b1][b2][b3].append(test_htable[x][y][z][0][0][3])
    max_vote = 0
    max_model_id = 0
    max_votes_list = []

    for model_id in votes_count:
        for b1 in votes_count[model_id]:
            for b2 in votes_count[model_id][b1]:
                for b3 in votes_count[model_id][b1][b2]:
                    if votes_count[model_id][b1][b2][b3] > max_vote:
                        max_vote = votes_count[model_id][b1][b2][b3]
                        max_model_id = model_id
                        max_votes_list = votes_lists[model_id][b1][b2][b3]
    # check if a match exists. i.e., if the number of max votes is above the threshold
    if max_vote >= threshold:
        return max_model_id, max_votes_list
    else:
        return None, []


def recognition_test(lines, full_htable, threshold, bin_size, model_lengths, model_names):
    identified_models_coord = {}
    identified_models_points = {}
    model_id = 0

    p_coord = [[], [], [], [], []]
    for ii in range(0, len(lines)):
        # fetch atom data record
        record = lines[ii].split()
        # Only read alpha carbon atoms
        if not lines[ii].startswith("ATOM"):
            continue
        if record[2] != "CA":
            continue
        # extract data from ATOM record
        p_coord[0].append(float(record[6]))
        p_coord[1].append(float(record[7]))
        p_coord[2].append(float(record[8]))
        p_coord[3].append(0)  # current identified model
        p_coord[4].append(0)  # identified model ID
    b_i = range(0, len(p_coord[0]))
    shuffle(b_i)
    for b1 in b_i:
        print(b1)
        x1 = p_coord[0][b1]
        y1 = p_coord[1][b1]
        z1 = p_coord[2][b1]
        p1 = Point3D(float(x1), float(y1), float(z1), b1)
        for b2 in b_i:
            if b1 != b2:  # for each unique point pair
                x2 = p_coord[0][b2]
                y2 = p_coord[1][b2]
                z2 = p_coord[2][b2]
                p2 = Point3D(float(x2), float(y2), float(z2), b2)

                for b3 in b_i:
                    if b1 != b3 and b2 != b3:  # for each unique point pair
                        x3 = p_coord[0][b3]
                        y3 = p_coord[1][b3]
                        z3 = p_coord[2][b3]
                        p3 = Point3D(float(x3), float(y3), float(z3), b3)

                        trans_info = calculate_transformation_parameters(p1, p2, p3, bin_size)
                        if trans_info is None:
                            # points are collinear, pick another combination
                            continue
                        test_htable = {}
                        for p_index in range(0, len(p_coord[0])):  # after picking the bases, loop through remaining points
                            p_coord[3][p_index] = -1
                            if p_index != b1:
                                point_x = p_coord[0][p_index]
                                point_y = p_coord[1][p_index]
                                point_z = p_coord[2][p_index]
                                point = Point3D(float(point_x), float(point_y), float(point_z), p_index)
                                new_point = transform_point3d(trans_info, point, bin_size)
                                test_htable = add_entry_to_hash_table(test_htable, new_point, b1, b2, b3, model_id, True)

                        matched_model_id, voting_point_ids = search_for_a_match(full_htable, test_htable, threshold, model_lengths)

                        if matched_model_id is not None:
                            voting_point_coordinates = [[], [], []]
                            voting_point_ids.append(b1)

                            for k in range(0, len(voting_point_ids)):
                                pid = voting_point_ids[k]
                                xv, yv, zv = p_coord[0][pid], p_coord[1][pid], p_coord[2][pid]
                                voting_point_coordinates[0].append(float(xv))
                                voting_point_coordinates[1].append(float(yv))
                                voting_point_coordinates[2].append(float(zv))

                                p_coord[3][pid] = int(matched_model_id)  # coloring for current model
                                p_coord[4][pid] = int(matched_model_id)  # all recognized models

                            already_identified = False
                            if matched_model_id in identified_models_points:
                                for i in range(0, len([matched_model_id])):
                                    if sorted(voting_point_ids) == sorted(identified_models_points[matched_model_id][i]):
                                        already_identified = True
                                        break

                            # if the base fragment and the new one are similar
                            if not already_identified:
                                if matched_model_id not in identified_models_points:
                                    # identified_models_coord[matched_model_id] = []
                                    identified_models_points[matched_model_id] = []
                                print voting_point_ids
                                fig = plt.figure()
                                ax = fig.add_subplot(111, projection='3d')
                                ax.scatter(p_coord[0], p_coord[1], p_coord[2], c=p_coord[3])
                                plt.title("Model " + str(model_names[matched_model_id]))
                                plt.show()
                                print("MATCH DETECTED")
                                print(len(voting_point_ids))
                                # identified_models_coord[matched_model_id].append(voting_point_coordinates)
                                identified_models_points[matched_model_id].append(voting_point_ids)

    print identified_models_points.keys()
    return identified_models_points.keys()


# MAIN FLOW #
threshold = 0.99
bin_size = 0.001
full_hash_table, model_lengths, model_names = start_offline_training(threshold, bin_size)


test_file = open('test scenes 3D/test_' + str(0) + '.pdb', 'r')
# test_file = open('test scenes 3D/1a0c.pdb', 'r')

lines = test_file.readlines()
test_file.close()

recognition_test(lines, full_hash_table, threshold, bin_size, model_lengths, model_names)
