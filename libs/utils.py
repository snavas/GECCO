from collections import Counter
import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def sortByDistanceToCentroid(group, centroid):
    #return np.sqrt(np.power(group[:, 0]-centroid[0], 2)+np.power(group[:, 1]-centroid[1], 2))
    #.sin(x[:, 1])WWZ
    distances = []
    for p in group:
        distances.append(np.sqrt(np.power(p[0]-centroid[0], 2) + np.power(p[1]-centroid[1], 2)))
    return distances

def getNearestPoint(group, centroid):
    #return np.sqrt(np.power(group[:, 0]-centroid[0], 2)+np.power(group[:, 1]-centroid[1], 2))
    #.sin(x[:, 1])WWZ
    nearestDistance = 100000000;
    nearestPoint = []
    for p in group:
        currentDistance = np.sqrt(np.power(p[0]-centroid[0], 2) + np.power(p[1]-centroid[1], 2))
        if currentDistance < nearestDistance:
            nearestPoint = [p[0], p[1], p[2]]
            #nearestPoint = p[2]
    return nearestPoint

def groupPointsbyLabels(points, labels):
    #print(points)
    #print(labels)
    n = len(Counter(labels).keys())  # equals to list(set(words))
    grouped = []
    # create empty array for N groups
    for i in range(0,n):
        grouped.append([])
    # fill the groups arrays
    for i in range(0, len(points)):
        grouped[labels[i]].append([points[i][0], points[i][1], points[i][2]])
    central = []
    for i in range(0, len(grouped)):
        x, y, index = zip(*grouped[i])
        centroid = (max(x) + min(x)) / 2., (max(y) + min(y)) / 2.
        #order = np.argsort(sortByDistanceToCentroid(grouped[i], centroid))
        #sorted_group = grouped[i][order]
        #central.append([sorted_group[i][0], sorted_group[i][1], sorted_group[i][2]])
        #central.append(center)
        central.append(getNearestPoint(grouped[i], centroid))
    #print("n= ", n, " - central.len= ", len(central))
    #print("##############")
    return central

def id_to_random_color(number):
    if number == -1:
        return (0,0,0)
    elif number == 0:
        return (31,120,180)
    elif number == 1:
        return(178,223,138)
    elif number == 2:
        return(51,160,44)
    elif number == 3:
        return(251,154,153)
    elif number == 4:
        return(227,26,28)
    elif number == 5:
        return(253,191,111)
    elif number == 6:
        return(255,127,0)
    elif number == 7:
        return(202,178,214)
    elif number == 8:
        return(106,61,154)
    elif number == 9:
        return (166,206,227)
    elif number == 10:
        return (31,120,180)
    elif number == 11:
        return(178,223,138)
    elif number == 12:
        return(51,160,44)
    elif number == 13:
        return(251,154,153)
    elif number == 14:
        return(227,26,28)
    elif number == 15:
        return(253,191,111)
    elif number == 16:
        return(255,127,0)
    elif number == 17:
        return(202,178,214)
    elif number == 18:
        return(106,61,154)
    else:
        return(255,255,255)
