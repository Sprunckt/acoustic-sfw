import numpy as np

threshold = 1e-2


def same_half_space(x, y, normal, origin):
    return (np.dot(x, normal) - origin)*(np.dot(y, normal) - origin) > 0


def find_order1(image_pos):
    norms = np.linalg.norm(image_pos, axis=1)
    min_dist_ind = np.argmin(norms)
    source_pos = image_pos[min_dist_ind]
    image_sources = image_pos[np.arange(len(image_pos)) != min_dist_ind]  # image sources (removed the source)
    n_image_src = len(image_sources)
    dist = np.linalg.norm(source_pos[np.newaxis, :] - image_sources, axis=1)  # distances to source
    order1 = np.empty([6, 3], dtype=float)
    wall_pos = np.empty([6, 3], dtype=float)
    sorted_dist_ind = np.argsort(dist)

    closest_src = image_sources[sorted_dist_ind[0]]  # image source that is closest to the source
    order1[0] = closest_src
    wall_pos[0] = (closest_src + source_pos) / 2.
    normal_vect, origins = np.zeros((6, 3)),  np.zeros(6)
    normal_vect[0] = closest_src - source_pos
    origins[0] = np.dot(wall_pos[0], normal_vect[0])

    it_dist, nb_found = 1, 1
    while it_dist < n_image_src and nb_found < 6:  # loop over the image sources until an order 1 is found
        found = True
        curr_source = image_sources[sorted_dist_ind[it_dist]]
        for k in range(nb_found):  # check if the point is in the same half-space as another order1 source
            if same_half_space(order1[k], curr_source, normal_vect[k], origins[k]):
                found = False
                break
        if found:
            order1[nb_found] = curr_source
            wall_pos[nb_found] = (source_pos + order1[nb_found]) / 2.
            normal_vect[nb_found] = order1[nb_found] - source_pos
            origins[nb_found] = np.dot(wall_pos[nb_found], normal_vect[nb_found])
            nb_found += 1

        it_dist += 1
    return order1[:nb_found], wall_pos[:nb_found]


if __name__ == "__main__":
    pass
