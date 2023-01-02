import numpy as np

def calculate_distance(db_result, vector, threshold):
    distance = float(threshold)
    idx = None
    dist = None
    try:
        for row in db_result:
            vec = np.fromstring(row[1][1:-1], dtype=float, sep=',')
            dist = np.dot(vec,vector)
            if dist > distance:
                idx = row[0]
    except Exception as error:
        print('Error: ' + str(error))
        return idx, dist