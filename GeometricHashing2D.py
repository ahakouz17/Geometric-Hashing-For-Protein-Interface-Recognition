# This code handles 2D model-based objects recognition using geometric hashing algorithm
# Authors: Asma Hakouz, Dalyah jamal, Dzenaida gicic
# 11/1/2018


import math
from datetime import datetime
import matplotlib.pyplot as plt


class TransformationParams:
    def __init__(self, angle, scale, offset_point):
        self.angle = angle
        self.scale_factor = scale
        self.offset = offset_point


class Point2D:
    def __init__(self, x, y, pid):
        self.x = x
        self.y = y
        self.id = pid


def calculate_transformation_parameters(point1, point2):
    # this function caclulates the transformation parameters for the bases pair (point1 and point2)
    # it returns a TransformationParams object which contains info about rotation angle and scale
    # factor
    swapped = False  # to keep track if the points were swapped or not

    x1, y1 = float(point1.x), float(point1.y)
    x2, y2 = float(point2.x), float(point2.y)

    # we need to scale out new reference coordinate system so that the distance between the two points (bases)
    # is 1
    scale = math.sqrt((x1-x2)**2 + (y1-y2)**2)

    # to unify the convention, make sure that the first point is always to the left of the second point
    # this will limit the rotation angle values to be in the range [-90, 90] degrees
    # if they're not arranged in that configuration, swap them and add +180/-180 after angle calculations
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # calculate the rotation angle, the angle is calculated assuming the positive direction
    # from the positive x-axis to the point (CCW).
    # check the boundaries where the tangent might go to infinity (+90, -90)
    if x1 == x2:
        # same point or 90 or -90 degrees
        if y1 == y2:
            # same point
            return None
        elif y1 < y2:
            # 90 angle CW
            angle = 90
        else:
            # -90 angle CCW
            angle = -90
    else:
        # handle the general case, using inverse tangent and slope
        angle = math.degrees(math.atan(float(y2 - y1) / (x2 - x1)))

    if swapped:
        if angle < 0:
            angle += 180
        else:
            angle += -180

    transformation = TransformationParams(angle, scale, point1)

    ''' UNCOMMENT FOR DEBUGGING
    # to debug the functionality, if we transform point one and two their new coordinates should
    # be (0, 0) and (1, 0) respectively
    new_p2 = transform_point2d(transformation, point2, 1)
    print("new x2: " + str(new_p2.x) + " , new y2: " + str(new_p2.y))
    '''
    return transformation


def transform_point2d(trans_info, p, bin_size):
    # this function transforms a point from the original to the new coordinate system using the
    # transformation details provided in transformation_info object
    # calculate the sint and cost for rotation matrix
    sin_t = math.sin(math.radians(-1 * trans_info.angle))
    cos_t = math.cos(math.radians(-1 * trans_info.angle))

    # point transformation
    # translation by subtracting the offset, and rotation by the rotation angle
    new_x = ((p.x - trans_info.offset.x) * cos_t - (p.y - trans_info.offset.y) * sin_t) / trans_info.scale_factor
    new_y = ((p.x - trans_info.offset.x) * sin_t + (p.y - trans_info.offset.y) * cos_t) / trans_info.scale_factor

    # due to quantization we have to decide in which bin the point land
    new_x = int(round(new_x / bin_size))
    new_y = int(round(new_y / bin_size))

    return Point2D(new_x, new_y, p.id)


def add_entry_to_hash_table(htable, point, b1, b2, model_id, include_id):
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

    # check if there is already an entry for the model coordinate inside the point.x, point.y entry
    # if not create a new empty entry
    if model_id not in htable[point.x][point.y]:
        htable[point.x][point.y][model_id] = []

    # add the entry in the x,y bin with model ID and bases pair
    if include_id:
        htable[point.x][point.y][model_id].append([b1, b2, point.id])
    else:
        htable[point.x][point.y][model_id].append([b1, b2])

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

            for model_id in cluster_rep_table[x][y]:
                for b1, b2 in cluster_rep_table[x][y][model_id]:
                    if model_id not in votes_table:
                        votes_table[model_id] = {}

                    if b1 not in votes_table[model_id]:
                        votes_table[model_id][b1] = {}

                    if b2 not in votes_table[model_id][b1]:
                        votes_table[model_id][b1][b2] = 0

                    votes_table[model_id][b1][b2] += (1.0 / model_lengths[model_id])

    max_vote = 0
    for model_id in votes_table:
        for b1 in votes_table[model_id].keys():
            for b2 in votes_table[model_id][b1]:
                if votes_table[model_id][b1][b2] > max_vote:
                    max_vote = votes_table[model_id][b1][b2]

    # check if a match exists. i.e., if the number of max votes is above the threshold
    if max_vote >= thresh:
        return True
    else:
        return False


def merge_with_full_htable(full_htable, cluster_htable):
    for x_sub in cluster_htable:
        for y_sub in cluster_htable[x_sub]:
            model = cluster_htable[x_sub][y_sub].keys()[0]
            if x_sub not in full_htable:
                full_htable[x_sub] = {}
            if y_sub not in full_htable[x_sub]:
                full_htable[x_sub][y_sub] = {}
            full_htable[x_sub][y_sub][model] = cluster_htable[x_sub][y_sub][model]
    return full_htable


def start_offline_training(num_of_models, training_threshold, bin_size):
    # this function is the offline training part where we loop through the models and produce the
    # full hash table. Also, a clustering step was added to eliminate similar models by clustering
    # them together and choosing only one model to be the cluster representative then add its hash
    # table to the full table
    print('Offline learning started...')
    beginning = datetime.now()
    print(beginning)
    model_lengths = [0] * num_of_models

    # initialization
    num_unclustered_models = num_of_models
    cluster_ids = {}
    model_id = 0
    full_htable = {}

    # we look for an unclustered fragment, pick it as the cluster representative, construct its hash table, add it to
    # the full table. then find all matches of this cluster and eliminate them
    for b1 in range(0, num_of_models):
        cluster_ids['model_' + str(b1)] = 0  # 0 means that this fragment is not clustered yet

    while num_unclustered_models > 0:
        model_id += 1
        for k in range(0, num_of_models):
            if cluster_ids['model_' + str(k)] == 0:  # find the first unclustered model
                cluster_ids['model_' + str(k)] = model_id
                cluster_htable = {}
                break

        model_file = open('models2D/model_' + str(k) + '.txt', 'r')
        lines = model_file.readlines()
        model_lengths[k] = len(lines)
        model_file.close()

        ''' creates the full geometric hash table of the model'''
        for b1 in range(0, len(lines)):
            x1, y1 = lines[b1].split()
            p1 = Point2D(float(x1), float(y1), b1)

            for b2 in range(0, len(lines)):
                x2, y2 = lines[b2].split()
                p2 = Point2D(float(x2), float(y2), b2)

                if b1 != b2:  # check that we're picking two unique points as bases
                    trans_info = calculate_transformation_parameters(p1, p2)

                    for p_index in range(0, len(lines)):  # after picking the bases, loop through the remaining points
                        if p_index != b1:  # m!=j
                            point_x, point_y = lines[p_index].split()
                            point = Point2D(float(point_x), float(point_y), p_index)
                            new_point = transform_point2d(trans_info, point, bin_size)
                            cluster_htable = add_entry_to_hash_table(cluster_htable, new_point, b1, b2, model_id, False)

        num_unclustered_models -= 1
        merge_with_full_htable(full_htable, cluster_htable)
        # FOR DEBUGGING - Printing the hash table
        '''for x in clusterDic.keys():
                print(str(x))
                for y in clusterDic[x].keys():
                        print("    " + str(y))
                        print("        " + str(clusterDic[x][y]))
        print('====================================')'''

        # cluster all matching models
        for n in range(k + 1, num_of_models):  # loop through the remaining models
            model_lengths[n] = len(lines) - 1
            if cluster_ids['model_' + str(n)] == 0:
                model_file = open('models2D/model_' + str(n) + '.txt', 'r')
                lines = model_file.readlines()
                model_file.close()
                matched = False

                for b1 in range(0, len(lines)):
                    x1, y1 = lines[b1].split()
                    p1 = Point2D(float(x1), float(y1), b1)

                    for b2 in range(0, len(lines)):
                        x2, y2 = lines[b2].split()
                        p2 = Point2D(float(x2), float(y2), b2)

                        if b1 != b2:  # check that we're picking two unique points as bases
                            trans_info = calculate_transformation_parameters(p1, p2)
                            model_htabel = {}

                            for p_index in range(0, len(lines)):# after picking the bases, loop through remaining points
                                if p_index != b1:
                                    point_x, point_y = lines[p_index].split()
                                    point = Point2D(float(point_x), float(point_y), p_index)
                                    new_point = transform_point2d(trans_info, point, bin_size)
                                    model_htabel = add_entry_to_hash_table(model_htabel, new_point, b1, b2, model_id, False)
                            matched = compare_for_clustering(cluster_htable, model_htabel, model_lengths, training_threshold)
                            # print(matched)
                            if matched:  # if the base fragment and the new one are similar
                                cluster_ids['model_' + str(n)] = model_id
                                num_unclustered_models -= 1
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
    return full_htable, model_lengths


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

            for model_id in full_htable[x][y]:
                for b1, b2 in full_htable[x][y][model_id]:
                    if model_id not in votes_count:
                        votes_count[model_id] = {}
                        votes_lists[model_id] = {}

                    if b1 not in votes_count[model_id]:
                        votes_count[model_id][b1] = {}
                        votes_lists[model_id][b1] = {}

                    if b2 not in votes_count[model_id][b1]:
                        votes_count[model_id][b1][b2] = 0
                        votes_lists[model_id][b1][b2] = []

                    votes_count[model_id][b1][b2] += (1.0 / model_lengths[model_id])
                    votes_lists[model_id][b1][b2].append(test_htable[x][y][0][0][2])

    max_vote = 0
    max_model_id = 0
    max_votes_list = []

    for model_id in votes_count:
        for b1 in votes_count[model_id].keys():
            for b2 in votes_count[model_id][b1]:
                if votes_count[model_id][b1][b2] > max_vote:
                    max_vote = votes_count[model_id][b1][b2]
                    max_model_id = model_id
                    max_votes_list = votes_lists[model_id][b1][b2]
    # check if a match exists. i.e., if the number of max votes is above the threshold
    print(max_vote)
    if max_vote >= threshold:
        # print(max_model_id)
        # print(max_votes_list)
        return max_model_id, max_votes_list
    else:
        return None, []


def recognition_test(lines, full_htable, threshold, bin_size, model_lengths):
    identified_models_coord = {}
    identified_models_points = {}
    model_id = 0

    p_coord = [[], [], [], []]
    for ii in range(0, len(lines)):
        x, y = lines[ii].split()
        p_coord[0].append(float(x))
        p_coord[1].append(float(y))
        p_coord[2].append(0)  # current identified model
        p_coord[3].append(0)  # identified model ID

    for b1 in range(0, len(lines)):
        x1 = p_coord[0][b1]
        y1 = p_coord[1][b1]
        p1 = Point2D(float(x1), float(y1), b1)

        for b2 in range(0, len(lines)):
            x2 = p_coord[0][b2]
            y2 = p_coord[1][b2]
            p2 = Point2D(float(x2), float(y2), b2)

            if b1 != b2:  # for each unique point pair
                trans_info = calculate_transformation_parameters(p1, p2)
                test_htable = {}
                for p_index in range(0, len(lines)):  # after picking the bases, loop through remaining points
                    p_coord[2][p_index] = 0
                    if p_index != b1:
                        point_x = p_coord[0][p_index]
                        point_y = p_coord[1][p_index]
                        point = Point2D(float(point_x), float(point_y), p_index)
                        new_point = transform_point2d(trans_info, point, bin_size)
                        test_htable = add_entry_to_hash_table(test_htable, new_point, b1, b2, model_id, True)

                matched_model_id, voting_point_ids = search_for_a_match(full_htable, test_htable, threshold, model_lengths)

                if matched_model_id is not None:
                    voting_point_coordinates = [[], []]
                    voting_point_ids.append(b1)
                    for k in range(0, len(voting_point_ids)):
                        pid = voting_point_ids[k]
                        xv, yv = lines[pid].split()
                        voting_point_coordinates[0].append(float(xv))
                        voting_point_coordinates[1].append(float(yv))
                        p_coord[2][pid] = int(matched_model_id)  # should be colored
                        p_coord[3][pid] = int(matched_model_id)  # should be colored

                    already_identified = False
                    if matched_model_id in identified_models_points:
                        for i in range(0, len(identified_models_points[matched_model_id])):
                            if sorted(voting_point_ids) == sorted(identified_models_points[matched_model_id][i]):
                                already_identified = True
                                break

                    # if the base fragment and the new one are similar
                    # TODO we have to check if it was detected by the same points in the same place before
                    if not already_identified:
                        plt.scatter(p_coord[0], p_coord[1], c=p_coord[2])
                        plt.title("Model " + str(matched_model_id))
                        plt.show()
                        if matched_model_id not in identified_models_points:
                            identified_models_coord[matched_model_id] = []
                            identified_models_points[matched_model_id] = []
                        identified_models_coord[matched_model_id].append(voting_point_coordinates)
                        identified_models_points[matched_model_id].append(voting_point_ids)

    print identified_models_coord.keys()
    return identified_models_coord.keys()


# MAIN FLOW #
num_of_models = 7
threshold = 0.75
bin_size = 0.01
full_hash_table, model_lengths = start_offline_training(num_of_models, threshold, bin_size)


test_file = open('test scenes/test_' + str(0) + '.txt', 'r')
lines = test_file.readlines()
test_file.close()

recognition_test(lines, full_hash_table, threshold, bin_size, model_lengths)