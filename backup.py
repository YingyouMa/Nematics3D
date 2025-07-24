# def visual_loops_genus(lines_origin, N, grid_g, 
#                        window_loop=21, N_ratio_loop=1.5, radius=2,
#                        if_lines_added=False, wrap=True, 
#                        color_list=[]):
    
#     '''

#     #! Color and grid_g seperately

#     Visualize loops using Mayavi.
#     Loops are colored according to their genus.

#     Given the director field n, visualize the disclination loops by the genus as the following:

#     - Detect the defects (defect_indices) in the director field by defect_detect(n, boundary=True)

#     - Find the thickened disclination loops and their genus by extract_loop(defect_indices, N)
#       It will provide grid_g, labelling each defect with the genus of the line that this defect belongs to.
    
#     - Sort the defects into different lines by lines = defect_connected(defect_indices, N)
#       Need this step to smoothen the lines.

#     - visual_loops_genus(lines, N, grid_g)

#     Parameters
#     ----------
#     lines : list of arrays
#             List of disclination lines, where each line is represented as an array of coordinates.
#             Usually provided by defect_connected()

#     N : int
#         Size of the grid in each dimension.

#     grid_g : numpy.ndarray, shape (N, N, N)
#              The defects indices and genus represented by grid of box.
#              N is the size of the 3D grid along each dimension.
#              If one point is 0, there is no defect here.
#              If one point is x (not zero), there is a defect here and the genus of this line is x-1.
#              Usually provided by extract_loop().

#     window_loop : int, optional
#                   Window length for smoothening the loops using Savitzky-Golay filter.
#                   Default is 21.

#     N_ratio_loop : float, optional
#                    Ratio to determine the number of points in the output smoothened loop.
#                    N_out = N_ratio_loop * len(loop).
#                    Default is 1.5.

#     radius : int, optional
#              Scale factor for visualizing the loops.
#              Default is 2.

#     if_lines_added : bool, optional
#                      Flag indicating whether midpoints have been added to disclination lines.
#                      Default is False.

#     wrap : bool, optional
#            Flag indicating whether to wrap the loops in visualization
#            Default is True.

#     color_list : array-like, optional, shape (M,)
#                  The color for each disclination loop. Each value belongs to [0,1]
#                  The colormap is derived by blue_red_in_white_bg() in the same module.
#                  0 for blue, 1 for red.
#                  If the length of color_list is 0, the loops are colored by genus.
#                  Default is [] (loops colored by genus)


#     Dependencies
#     ------------
#     - mayavi: 4.7.4
#     - scipy: 1.7.3
#     - numpy: 1.22.0
#     '''

#     from mayavi import mlab
#     from scipy.spatial.distance import cdist

#     # If midpoints are not added to disclination lines, add them
#     lines = 1 * lines_origin
#     if if_lines_added == False:
#         for i, line in enumerate(lines):
#             lines[i] = add_mid_points_disclination(line)

#     # Initialize an array to store genus information for each line
#     genus_lines = np.zeros(len(lines))

#     # Get coordinates of defects in the grid
#     defect_g = np.array(np.nonzero(grid_g)).T

#     # Find genus for each line based on the proximity of its head to defect points
#     for i, line in enumerate(lines):
#         head = line[0]%N
#         dist = cdist([head], defect_g)
#         if np.min(dist) > 1:
#             genus_lines[i] = -1
#         else:
#             genus_lines[i] = grid_g[tuple(defect_g[np.argmin(dist)])] - 1

#     lines = np.array(lines, dtype=object)
#     loops = lines[genus_lines>-1]
#     genus_loops = genus_lines[genus_lines > -1]

#     mlab.figure(bgcolor=(1,1,1))

#     colormap = blue_red_in_white_bg()

#     for i, loop in enumerate(loops):
#         genus = genus_loops[i]
#         if len(color_list) == 0:
#             if genus == 0:
#                 color = (0,0,1)
#             elif genus == 1:
#                 color = (1,0,0)
#             elif genus > 1:
#                 color= (0,0,0)
#         else:
#             color = colormap[int(color_list[i]*510)]
#         loop = smoothen_line(loop, window_length=window_loop, mode='wrap', 
#                              N_out=int(N_ratio_loop*len(loop)))
        
#         if wrap == True:
#             loop = loop%N

#         mlab.points3d(*(loop.T), scale_factor=radius, color=tuple(color))








# def check_find_old(defect_here, defect_group, defect_box, N):

#     from scipy.spatial.distance import cdist
    
#     defect_plane_axis = np.where( ( defect_here % 1) == 0 )[0][0]

#     if_find = False

#     dist = cdist(defect_group, defect_box, metric='sqeuclidean')
#     defect_where = np.where(dist == 0.5)[1]
#     if len(defect_where) == 0:
#         defect_where = np.where(dist == 1)[1]
#         if len(defect_where) > 0:
#             for item in defect_where:
#                 defect_ordinal_next = item
#                 defect_next = defect_box[defect_ordinal_next]
#                 defect_diff = defect_next - defect_here
#                 if defect_plane_axis == (np.where( (np.abs(defect_diff) == 1) + (np.abs(defect_diff) == N-1) ))[0][0]:
#                     if_find = True
#                     break
#     else:
#         if_find = True
#         defect_ordinal_next = defect_where[0]
#         defect_next = defect_box[defect_ordinal_next]
        
#     if if_find == False:
#         defect_ordinal_next, defect_next = None, None
        
#     return if_find, defect_ordinal_next, defect_next

# def defect_connected_old(defect_indices, N, print_time=False, print_per=1000):
    
#     #! N_index for 3 different axes

#     """
#     Classify defects into different lines.

#     Parameters
#     ----------
#     defect_indices : numpy array, num_defects x 3
#                      Represents the locations of defects in the grid.
#                      For each location, there must be one integer (the index of plane) and two half-integers (the center of the loop on that plane)
#                      This is usually given by defect_detect()

    
    
#     """

#     #! It only works with boundary=True in defect_detect() if defects cross walls

#     index = 0
#     defect_num= len(defect_indices)
#     defect_left_num = defect_num
    
#     lines = []
    
#     check0 = defect_indices[:,0] < int(N/2)
#     check1 = defect_indices[:,1] < int(N/2)
#     check2 = defect_indices[:,2] < int(N/2)
    
#     defect_box = []
#     defect_box.append( defect_indices[ np.where( check0 * check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( check0 * ~check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( check0 * check1 * ~check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * ~check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * check1 * ~check2 ) ] )
#     defect_box.append( defect_indices[ np.where( check0 * ~check1 * ~check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * ~check1 * ~check2 ) ] )
    
#     start = time.time()
#     start_here = time.time()

#     while defect_left_num > 0:
        
#         loop_here = np.zeros( ( len(defect_indices),3 ) )
#         defect_ordinal_next = 0
#         cross_wall = np.array([0,0,0])
        
#         box_index = next((i for i, box in enumerate(defect_box) if len(box)>0), None)
#         defect_box_here = defect_box[box_index]
#         loop_here[0] = defect_box_here[0]
        
#         index_here = 0
        
#         while True:
            
#             defect_ordinal = defect_ordinal_next
#             defect_box_here = 1*defect_box[box_index]
#             defect_here = defect_box_here[defect_ordinal]
#             defect_box[box_index] = np.vstack(( defect_box_here[:defect_ordinal], defect_box_here[defect_ordinal+1:] ))
#             defect_box_here = 1*defect_box[box_index]
            
#             defect_group = np.array([defect_here])

#             if_find = False
            
#             if len(defect_box_here) > 0:
#                 if_find, defect_ordinal_next, defect_next = check_find_old(defect_here, defect_group, defect_box_here, N)

#             if if_find == False or len(defect_box_here) == 0:
                
#                 defect_box_all = np.concatenate([box for box in defect_box])
            
#                 bound_0 = np.where( (defect_here==0) + (defect_here==N-1) )[0]
#                 if len(bound_0) > 0:
#                     defect_bound = 1*defect_here
#                     defect_bound[bound_0[0]] = trans_period(defect_bound[bound_0[0]],N)
#                     defect_group = np.concatenate([defect_group, [defect_bound]])
#                 bound_1 = np.where(defect_here==(N-0.5))[0]
#                 for bound in bound_1:
#                     defect_bound = 1*defect_here
#                     defect_bound[bound] = -0.5
#                     defect_group = np.concatenate([defect_group, [defect_bound]])
#                 if len(bound_0) > 0 and len(bound_1) > 0:
#                     for bound in bound_1:
#                         defect_bound = 1*defect_here
#                         defect_bound[bound] = -0.5
#                         defect_bound[bound_0[0]] = trans_period(defect_bound[bound_0[0]],N)
#                         defect_group = np.concatenate([defect_group, [defect_bound]])
                        
#                 if_find, defect_ordinal_next, defect_next = check_find_old(defect_here, defect_group, defect_box_all, N)
#                 if if_find == True:
#                     box_index, defect_ordinal_next = find_box(defect_ordinal_next, [len(term) for term in defect_box])
                
#             if if_find == True:
#                 defect_diff = defect_next - defect_here
#                 cross_wall_here = np.trunc( defect_diff / (N-10) )
#                 cross_wall = cross_wall - cross_wall_here
#                 defect_next = defect_next + cross_wall * N
#                 loop_here[index_here+1] = defect_next
#                 index += 1
#                 index_here += 1
#                 if print_time == True:
#                     if index % print_per == 0:
#                         print(f'{index}/{defect_num} = {round(index/defect_num*100,2)}%, {round(time.time()-start_here,2)}s  ',
#                             f'{round(time.time()-start,2)}s in total' )
#                         start_here= time.time()
#             else:
#                 zero_loc = np.where(np.all([0,0,0] == loop_here, axis=1))[0]
#                 if len(zero_loc) > 0:
#                     loop_here = loop_here[:zero_loc[0]]
#                 lines.append(loop_here)
#                 defect_left_num = 0
#                 for term in defect_box:
#                     defect_left_num += len(term)
#                 break
    
#     return lines

# def lines_connect_closed(lines_list, box_size):

#     from .field import get_ghost_point
#     from itertools import product

#     closed_list = []
#     wait_list = []
#     closed_no_list=[]

#     for line in lines_list:
#         if is_defects_connected( line[0], line[-1]):
#             closed_list.append(line)
#         else:
#             point1 = get_ghost_point(line[0]%box_size, box_size)
#             point2 = get_ghost_point(line[-1]%box_size, box_size)
#             wait_list.append([line, [point1, point2]])

#     while len(wait_list)>1:
#         wait_line = wait_list[0]
#         for count in (0, 1):
#             for i in range(len(wait_list)-1,0,-1):
#                 test_line = wait_list[i]
#                 point_start = wait_line[1][count]
#                 for j in (0,1):
#                     if_find = False
#                     for point1, point2 in product(point_start, test_line[1][j]):
#                         #print(point1, point2)
#                         if is_defects_connected(point1, point2):
#                             print('found')
#                             if_find = True
#                             cross_box = (wait_line[0][-count] - test_line[0][-j])//box_size
#                             to_add = test_line[0] + cross_box * box_size
#                             if count==0 and j==1:
#                                 wait_list[0][0] = np.vstack([to_add, wait_line[0]])
#                                 wait_list[0][1][0] = test_line[1][0]
#                             if count==1 and j==0:
#                                 wait_list[0][0] = np.vstack([wait_line[0], to_add])
#                                 wait_list[0][1][1] = test_line[1][1]    
#                             wait_list.pop(i)
#                             break
#                     if if_find == True:
#                         break

# def defect_classify_into_lines_init(defect_indices, box_size, print_time=False, print_per=1000):
    
#     #! It only works with boundary=True in defect_detect() if defects cross walls
#     #! box_size, when defects are not throughout the whole box but gather in small region
#     #! set box_size to be optional value

#     """
#     The first step to classify defects into different lines.

#     Parameters
#     ----------
#     defect_indices : numpy array, (num_defects, 3)
#                      Represents the locations of defects in the grid.
#                      For each location, there must be one integer (the index of plane) and two half-integers (the center of the loop on that plane)
#                      This is usually given by defect_detect()

#     box_size : float or numpy of three floats
#                The largest index of the entire box in each dimension.
#                Used for periodic boundary condition and sub-box-division for acceleration.
#                If box_size is one integer as x, it is interprepted as (x,x,x).

#     Returns
#     -------
#     defect_sorted : (num_defects, 5)
#                     The defects classified into different disclination lines.
#                     For each defect, the first three values are the indices.
#                     The fourth one is the index of the line that this defect belongs to.
#                     The fifth one is the index of this defect in the line that this defect belongs to.

#     Dependencies
#     ------------
#     - scipy: 1.7.3
#     - numpy: 1.22.0 

#     Called by
#     ---------
#     - Disclination_line  

#     Internal functions
#     -------------------------
#     - trans_period()
#     - check_find()
#     - find_box()
#     """

#     # Begin internal functions
#     # ------------------------

#     def trans_period(n, N):
#         if n == 0:
#             return N
#         elif n == N-1:
#             return -1
    
#     def check_find(defect_here, defect_reservoir, defect_group=0,  box_size=0):
#         ''' 
#         To find if defect_group contains one defect neighboring to defect_here.
#         The periodic boundary condition could be put into consideration.
#         If several different neighboring defects are found in defect_reservoir, only one of them will be returned.

#         Parameters
#         ----------
#         defect_here : numpy array, (3,)
#                       the indices of the defect, provided by defect_detect().
#                       One of the index must be integer, representing the layer,
#                       and the other two indices must be half-integer, representing the center of one pixel in this layer
#                       Supposing defect_here = (layer, center1, center2), where layer is integer while center1 and center2 are half-integers,
#                       the set of all the possible neighboring defects is
#                       (layer+-1,     center1,        center2)
#                       (layer+-0.5,   center1+-0.5,   center2)
#                       (layer+-0.5,   center1,        center2+-0.5)
#                       here +- means plusminus, and the order is unneccessary as (+,+), (-,+), (+,-), (-,-) are all possible
#                       so there are 2+4+4=10 possible neighboring defects
#                       We use scipy.spatial.distance.cdist to find the neighboring defects, with metric='sqeuclidean'.
#                       If there exist neighboring defects, the distance will be 0.5**2 + 0.5**2 = 0.5 or 1.
#                       If the distance is 0.5, it must be the neighboring defect because (layer, conter1+-0.5, center2+-0.5) is not possible in defect_reservoir.
#                       If the distance is 1, we should check if the difference comes from the layer, as (layer, conter1+-1, center2) is possible in defect_reservoir but it's not neighboring defect.

#         defect_reservoir : numpy array, N x 3,
#                            The indices of other defects.
#                            The function will try to find if there is one defect in defect_reservoir such that the defect is neighboring to one of the defects in defect_group.
#                            Provided by defect_detect().
#                            The indices of each defect should have the same structure with defect_here, as one integer and two half-integers.

#         defect_group : numpy array, M x 3,
#                        The indices of the current defect and its ghost defects. In total there are M defects.
#                        The ghost defects come from the periodic boundary condition.     
#                        If there is no periodic boundary condition or defect_here is not near the boundary,
#                        defect_group should only contain defect_here
#                        Default is 0, where defect_group is interprepted as [defect_here]

#         box_size : int or numpy array of three ints
#                    The largest index of the entire box in each dimension.
#                    Used for periodic boundary condition.
#                    If box_size is one integer as x, it is interprepted as (x,x,x).
#                    Default is 0, where the periodic boundary condition is not considered

#         Return
#         ------
#         if_find : bool
#                 Whether find one neighboring defect

#         defect_next :  array of three ints:
#                     The indices of the neighboring defect

#         defect_ordinal_next : int
#                             The ordinal of the neighboring defect in defect_reservoir,
#                             such that defect_reservoir[defect_ordinal_next] = defect_next
#         '''

#         from scipy.spatial.distance import cdist
        
#         if len(np.shape([box_size])) == 1:
#             box_size = (box_size, box_size, box_size)
#         if len(np.shape([defect_group])) == 1:
#             defect_group = [defect_here]
        
#         defect_plane_axis = np.where( ( defect_here % 1) == 0 )[0][0] # find the integer index, as the layer

#         if_find = False

#         dist = cdist(defect_group, defect_reservoir, metric='sqeuclidean') # find the distance between each defect in defect_group and each defect in defect_reservoir
#         defect_where = np.where(dist == 0.5)[1] # If there exist dist==0.5 between one defect in defect_group and one defect in defect_reservoir, this defect in defect_reservoir is the neighboring defect
#         if len(defect_where) == 0: # If there is no dist==0.5, check if there is dist==1.
#             defect_where = np.where(dist == 1)[1] # If so, make sure the difference comes from the layer (from layer to layer+-1)
#             if len(defect_where) > 0:
#                 for item in defect_where:
#                     defect_ordinal_next = item
#                     defect_next = defect_reservoir[defect_ordinal_next]
#                     defect_diff = defect_next - defect_here
#                     # check in which axis the difference is 1 or size-1, where size is the largest index in the axis of layer
#                     # if the periodic boundary condition is not considered, size-1 will be -1, which will be automatically omiited due to np.abs()
#                     if defect_plane_axis == (np.where( (np.abs(defect_diff) == 1) + (np.abs(defect_diff) == box_size[defect_plane_axis]-1) ))[0][0]:
#                         if_find = True
#                         break
#         else:
#             if_find = True
#             defect_ordinal_next = defect_where[0]
#             defect_next = defect_reservoir[defect_ordinal_next]
            
#         if if_find == False:
#             defect_ordinal_next, defect_next = None, None
            
#         return if_find, defect_next, defect_ordinal_next

#     def find_box(value, length_list):
        
#         cumulative_sum = 0
#         position = 0

#         for i, num in enumerate(length_list):
#             cumulative_sum += num
#             if cumulative_sum-1 >= value:
#                 position = i
#                 break

#         index = value - (cumulative_sum - length_list[position])
#         return (position, index)


#     # Begin main function
#     # -------------------

#     from scipy.spatial.distance import cdist

#     if len(np.shape([box_size])) == 1:
#         box_size = np.array([box_size, box_size, box_size])
#     else:
#         box_size = np.array(box_size)
    
#     defect_indices = np.array(defect_indices)
#     defect_num = len(defect_indices)
#     defect_left_num = defect_num # the amount of unclassified defects. Initially it is the number of all defects

#     # We start each disclination line at the wall, so that we firstly select the defects at the wall as the start point.
#     # If the periodic boundary condition is considered, there might be no need to worry about it.
#     # To the opposite, if the periodic boundary condition is NOT considered, the disclination line must start at the wall,
#     # because here the cross line does NOT move back to the start point.
#     # Let's elaborate it. Imagine a line crossing the wall. If we do NOT start at the wall but start in the middle, the line will end at the wall.
#     # The rest of the line will turn to be another line. In other words, here the line will be splitted into two lines.

#     if_wall = (defect_indices[:,0]==0) | \
#               (defect_indices[:,1]==0) | \
#               (defect_indices[:,2]==0) | \
#               (defect_indices[:,0]==box_size[0]-1) | \
#               (defect_indices[:,1]==box_size[1]-1) | \
#               (defect_indices[:,2]==box_size[2]-1)

#     # For each defect, except the three coorinates, let's give it some other properties for convenience:
#     # the index of defects through all the defects
#     # if it is at the wall 

#     # In the output, we will still have the coordinates of all defects (sorted by lines), but there will also be:
#     # the index of the line that this defect belongs to
#     # the index of the defect within the line

#     defect_indices = np.hstack([defect_indices, np.zeros((defect_num,2))])
#     defect_indices[..., -2] = np.arange(defect_num)
#     defect_indices[if_wall, -1] = True

#     defect_sorted = np.zeros((defect_num, 5))
#     defect_sorted[..., -2] = -1
#     defect_sorted[..., -1] = -1

#     # to divide the defects into 8 subboxes to accelerate
#     check0 = defect_indices[:,0] < int(box_size[0]/2)
#     check1 = defect_indices[:,1] < int(box_size[1]/2)
#     check2 = defect_indices[:,2] < int(box_size[2]/2)

#     defect_box = []
#     defect_box.append( defect_indices[ np.where( check0 * check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( check0 * ~check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( check0 * check1 * ~check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * ~check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * check1 * ~check2 ) ] )
#     defect_box.append( defect_indices[ np.where( check0 * ~check1 * ~check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * ~check1 * ~check2 ) ] )
    
#     start = time.time()
#     start_here = time.time()

#     # loop when there are still unclassfied defects
#     index_line = -1 # representing the index of the new line
#     index = -1 # how many defects have been classfied
#     while defect_left_num > 0:
        
#         # to start to find a new discliantion line
#         cross_wall = np.array([0,0,0]) # the array recording that how many time have the line crossed wach wall due to periodic boudnary condition
#         index_here = 0 # representing the index of the next defect in the new line
#         index_line += 1
#         index += 1

#         # To see if there are still defects at the wall.
#         # If so, start the new line at such defect.
#         box_index = next((i for i, box in enumerate(defect_box) if np.sum(box[..., -1])>0), None)
        
#         # defect_box_here is the subbox containing the start defect
#         # defect_ordinal_next is the index of the new found defect in the subbox
#         if box_index != None:
#             defect_box_here = defect_box[box_index]
#             defect_ordinal_next = np.argmax(defect_box_here[..., -1]>0)
#         else:
#             box_index = next((i for i, box in enumerate(defect_box) if len(box)>0), None) # select the box which still contains unclassfied defects
#             defect_box_here = defect_box[box_index]
#             defect_ordinal_next = 0

#         defect_start = defect_box_here[defect_ordinal_next] # the defect that start a new line
#         defect_sorted[index, :3] = defect_start[:3]
#         defect_sorted[index, 3] = index_line
#         defect_sorted[index, 4] = 0

#         # Once start a line, try to find neighboring defects until there is no any
#         while True:
            
#             # update the defect and subbox in the loop
#             defect_ordinal = defect_ordinal_next
#             defect_box_here = 1*defect_box[box_index]
#             defect_here = defect_box_here[defect_ordinal]
#             defect_here_loc = defect_here[:3]
#             defect_box[box_index] = np.vstack(( defect_box_here[:defect_ordinal], defect_box_here[defect_ordinal+1:] ))
#             defect_box_here = 1*defect_box[box_index]
            
#             # Array of the locations of the current defect and its ghost defects coming from periodic boudnary
#             # Initially there is no ghost defect since we are searching within the subbox
#             defect_group = np.array([defect_here_loc])

#             if_find = False

#             defect_plane_axis = np.where( ( defect_here_loc % 1) == 0 )[0][0]
#             layer = defect_here_loc[defect_plane_axis]
              
#             # At first, try to find neighboring defect in the current subbox.
#             # Thus the periodic boundary condition is ignored here.
#             if len(defect_box_here) > 0:
#                 if_find, defect_next_loc, defect_ordinal_next = check_find(defect_here_loc, defect_box_here[:,:3])

#             # If there is no neighboring defect in the current subbox, expand the searching area to the entire box
#             if if_find == False or len(defect_box_here) == 0:
                
#                 defect_box_all = np.concatenate([box for box in defect_box])

#                 # Now let's consider the periodic boundary condition
#                 # We need to generate ghost defects if the defect is on the boundary
#                 if layer==0 or layer==box_size[defect_plane_axis]-1:
#                     defect_bound = 1*defect_here_loc
#                     defect_bound[defect_plane_axis] = trans_period(layer, box_size[defect_plane_axis])
#                     defect_group = np.concatenate([defect_group, [defect_bound]])
#                 bound_half = np.where(defect_here_loc==(box_size-0.5))[0]
#                 if len(bound_half) > 0:
#                     for bound in bound_half:
#                         defect_bound = 1*defect_here_loc
#                         defect_bound[bound] = -0.5
#                         defect_group = np.concatenate([defect_group, [defect_bound]])
#                     if layer==0 or layer==box_size[defect_plane_axis]-1:
#                         for bound in bound_half:
#                             defect_bound = 1*defect_here_loc
#                             defect_bound[bound] = -0.5
#                             defect_bound[defect_plane_axis] = trans_period(layer, box_size[defect_plane_axis])
#                             defect_group = np.concatenate([defect_group, [defect_bound]])
                        
#                 if_find, defect_next_loc, defect_ordinal_next = check_find(defect_here_loc, defect_box_all[:,:3], defect_group=defect_group, box_size=box_size)
#                 if if_find == True:
#                     box_index, defect_ordinal_next = find_box(defect_ordinal_next, [len(term) for term in defect_box])
                
#             if if_find == True:
#                 # If we find the next defect, store the data
#                 defect_diff = defect_next_loc - defect_here_loc
#                 cross_wall_here = np.trunc( defect_diff / (box_size-10) ) #! this 10
#                 cross_wall = cross_wall - cross_wall_here
#                 defect_next_loc = defect_next_loc + cross_wall * box_size
#                 index_here += 1
#                 index += 1
#                 defect_sorted[index, :3] = defect_next_loc
#                 defect_sorted[index, -2] = index_line
#                 defect_sorted[index, -1] = index_here
#                 defect_box_here = defect_box[box_index]
#                 if print_time == True:
#                     if index % print_per == 0:
#                         print(f'{index}/{defect_num} = {round(index/defect_num*100,2)}%, {round(time.time()-start_here,2)}s  ',
#                             f'{round(time.time()-start,2)}s in total' )
#                         start_here= time.time()
#             else:
#                 defect_left_num = 0
#                 for term in defect_box:
#                     defect_left_num += len(term)
#                 break
    
#     return defect_sorted, box_size


# def extract_lines(defect_sorted, box_size, is_add_mid=True):
#     '''
#     Extract the disclination lines into several instancies of class Disclination_line

#     Parameters
#     ----------
#     defect_sorted : numpy array, (num_defects, >=4)
#                     The information of each defect.
#                     For each defect, the first three values are the indices.
#                     The fourth one is the index of the line that this defect belongs to.

#     Returns
#     -------
#     lines : list,
#             The list of all the lines.
#             Each line is an instance of Disclination_line.
#     '''

#     from .classes.disclination_line import DisclinationLine

#     defect_sorted = defect_sorted[:, :4]

#     line_start = np.array(np.where(defect_sorted[1:,3] - defect_sorted[:-1,3] == 1)[0]) + 1
#     line_start = np.concatenate([[0], line_start])
#     line_start = np.concatenate([line_start, [np.shape(defect_sorted)[0]]])

#     line_num = int(defect_sorted[-1, 3] + 1)
#     lines = []
#     for i in range(line_num):
#         lines.append(DisclinationLine(defect_sorted[ line_start[i]:line_start[i+1] , :3],
#                                       box_size,
#                                       is_add_mid=is_add_mid))

#     return lines


# def defect_classify_into_lines1(defect_indices, box_size, print_time=False, print_per=1000):
#     '''
#     To classify defects into different lines.

#     Parameters
#     ----------
#     To read the document in defect_classify_into_lines_init() please.

#     Returns
#     -------
#     lines : list,
#             The list of all the lines.
#             Each line is an instance of Disclination_line.
#     '''
#     defect_sorted, box_size = defect_classify_into_lines_init(defect_indices, box_size, 
#                                                               print_time=print_time, print_per=print_per)
#     lines = extract_lines(defect_sorted, box_size)

#     return lines

# def visual_disclinations(lines, N, N_index=1, min_length=30, radius=0.5,
#                          window_cross=33, window_loop=21,
#                          N_ratio_cross=1, N_ratio_loop=1.5,
#                          loop_color=(0,0,0), cross_color=None,
#                          wrap=True, if_lines_added=False,
#                          new_figure=True, bgcolor=(1,1,1)):
#     #! plot3d to make lines as tubes
#     #! axes
#     #! N_index
#     #! different size, and f p p
#     '''
#     Visualize disclination lines in 3D using Mayavi.

#     Parameters
#     ----------
#     lines : list of arrays
#             List of disclination lines, where each line is represented as an array of coordinates.

#     N_index : int or array of 3 ints
#               Size of the cubic 3D grid in each dimension.
#               If only one int x is input, it is interpreted as (x,x,x)

#     min_length : int, optional
#                  Minimum length of disclination loops to consider. 
#                  Length calculated as the number of points conposing the loop-type disclinations.
#                  Default is 30.

#     radius : float, optional
#              Radius of the points representing the disclination lines in the visualization. 
#              Default is 0.5.

#     window_cross : int, optional
#                    Window length for Savitzky-Golay filter for cross-type disclinations. 
#                    Default is 33.

#     window_loop : int, optional
#                   Window length for Savitzky-Golay filter for loop-type disclinations. 
#                   Default is 21.

#     N_ratio_cross : float, optional
#                     Ratio to compute the number of points in the output smoothened line for crosses.
#                     N_out = N_ratio_cross * num_points_in_cross
#                     Default is 1.

#     N_ratio_loop : float, optional
#                    Ratio to compute the number of points in the output smoothened line for loops.
#                    N_out = N_ratio_loop * num_points_in_loop
#                    Default is 1.5.

#     loop_color : array like, shape (3,)  or None, optional
#                  The color of visualized loops in RGB.
#                  If None, the loops will follow same colormap with crosses.
#                  Default is (0,0,0)

#     wrap : bool, optional
#            If wrap the lines with periodic boundary condition.
#            Default is True

#     if_lines_added : bool, optional
#                      If the lines have already been added the mid-points
#                      Default is False.

#     new_figure : bool, optional
#                  If True, create a new figure for the plot. 
#                  Default is True.

#     bgcolor : array of three floats, optional
#               Background color of the plot in RGB.
#               Default is (0, 0, 0), white.

#     Returns
#     -------
#     None

#     Dependencies
#     ------------
#     - mayavi: 4.7.4
#     - numpy: 1.22.0

#     Functions in same module
#     ------------------------
#     - add_mid_points_disclination: Add mid-points to disclination lines.
#     - smoothen_line: Smoothen disclination lines.
#     - sample_far: Create a sequence where each number is trying to be far away from previous numbers.
#     - blue_red_in_white_bg: A colormap based on blue-red while dinstinct on white backgroud.    
#     '''

#     from mayavi import mlab

#     '''
#     if len(np.shape([N_index])) == 1:
#         N_index = (N_index, N_index, N_index)
#     '''

#     # Filter out short disclination lines
#     lines = np.array(lines, dtype=object)
#     lines = lines[ [len(line)>=min_length for line in lines] ] 

#     # Add mid-points to each disclination line
#     if if_lines_added == False:
#         for i, line in enumerate(lines):
#             lines[i] = add_mid_points_disclination(line)

#     # Separate lines into loops and crosses based on end-to-end distance and smoothen them.
#     loops = []
#     crosses = []
#     for i, line in enumerate(lines):
#         end_to_end = np.sum( (line[-1]-line[0])**2, axis=-1 )
#         if end_to_end > 2:
#             cross = smoothen_line(line, window_length=window_cross, 
#                                     N_out=int(N_ratio_cross * len(line)))
#             crosses.append(cross)
#         else:
#             loop = smoothen_line(line, window_length=window_loop, mode='wrap', 
#                                     N_out=int(N_ratio_loop*len(line)))
#             loops.append(loop)

#     # wrap the discliantions with periodic boundary conditions
#     if wrap == True:
#         loops = np.array(loops, dtype=object)%N
#         crosses = np.array(crosses, dtype=object)%N

#     # Generate a new figure with the given background color if needed
#     if new_figure==True:
#         mlab.figure(bgcolor=bgcolor) 

#     # Generate colormap
#     colormap = blue_red_in_white_bg()

#     if cross_color == None:
#         # Sort crosses by length for better visualization if the color is not set
#         crosses = np.array(crosses, dtype=object)
#         cross_length = [len(cross) for cross in crosses]
#         crosses = crosses[np.argsort(cross_length)[-1::-1]]
#     if loop_color == None:
#         loops = np.array(loops, dtype=object)
#         loop_length = [len(loop) for loop in loops]
#         loops = loops[np.argsort(loop_length)[-1::-1]]
#     if cross_color == None and loop_color == None:
#         color_index = ( sample_far(len(lines)) * 510 ).astype(int)
#         for i,cross in enumerate(crosses):
#             mlab.points3d(*(cross.T), scale_factor=radius, color=tuple(colormap[color_index[i]])) 
#         for j, loop in enumerate(loops):
#             mlab.points3d(*(loop.T), scale_factor=radius, color=tuple(colormap[color_index[len(crosses)+j]]))
#     elif cross_color != None and loop_color == None:
#         color_index = ( sample_far(len(loops)) * 510 ).astype(int)
#         for i,cross in enumerate(crosses):
#             mlab.points3d(*(cross.T), scale_factor=radius, color=cross_color) 
#         for j, loop in enumerate(loops):
#             mlab.points3d(*(loop.T), scale_factor=radius, color=tuple(colormap[color_index[j]]))
#     elif cross_color == None and loop_color != None:
#         color_index = ( sample_far(len(crosses)) * 510 ).astype(int)
#         for i,cross in enumerate(crosses):
#             print(i)
#             # return cross
#             mlab.points3d(*(cross.T), scale_factor=radius, color=tuple(colormap[color_index[i]])) 
#             #mlab.plot3d(*(cross.T), tube_radius=radius, color=tuple(colormap[color_index[i]]))
#         for j, loop in enumerate(loops):
#             mlab.points3d(*(loop.T), scale_factor=radius, color=loop_color)
#     else:
#         for i,cross in enumerate(crosses):
#             mlab.points3d(*(cross.T), scale_factor=radius, color=cross_color) 
#             #mlab.plot3d(*(cross.T), tube_radius=radius, color=cross_color)
#         for j, loop in enumerate(loops):
#             mlab.points3d(*(loop.T), scale_factor=radius, color=loop_color)  

# ---------------------------------------
# Functions which are not currently used.
# Just for back up.
# ---------------------------------------


# def find_defect_n_old(defect_indices, size=0):
#     #! defect_indices half integer
#     '''
#     To find the directors enclosing the defects.
#     For example, the defect locates at (10, 5.5, 14.5) in the unit of indices
#     the directors enclosing the defects would locate at:
#     (10, 5, 14)
#     (10, 5, 15)
#     (10, 6, 14)
#     (10, 6, 15)

#     Parameters
#     ----------
#     defect_indices : array, (defect_num, 3)
#                      The array that includes all the indices of defects.
#                      For each defect, one of the indices should be integer and the rest should be half-integer
#                      Usually defect_indices are generated by defect_defect() in this module

#     size : int, or array of three ints
#            The size of the box, which is used to delete the directors outsize the box.
#            Directors outsize the box may appear when the defect is near the wall and periodic boundary condition is applied.
#            Default is 0, interprepted as do not delete any directors  

#     Returns
#     -------
#     defect_n : array, (4*defect_num, 3) 
#                The indices of directors enclosing the defects

#     Dependencies
#     ------------
#     - NumPy: 1.22.0               
#     '''

#     if len(np.shape([size])) == 1:
#         size = np.array([size]*3)
#     else:
#         size = np.array(size)
#     # delete all the defects which will generate directors out of the box
#     if np.sum(size) != 0:
#         defect_indices = defect_indices[np.where(np.all(defect_indices<=(size-1), axis=1))]

#     defect_num = np.shape(defect_indices)[0]

#     defect_n = np.zeros((4*defect_num,3))

#     defect_n0 = np.zeros((defect_num,3))
#     defect_n1 = np.zeros((defect_num,3))
#     defect_n2 = np.zeros((defect_num,3))
#     defect_n3 = np.zeros((defect_num,3))

#     # find the layer of each defect (where the index is integer)
#     cond_layer = defect_indices%1==0

#     # the layer is unchanged
#     defect_n0[cond_layer] = defect_indices[cond_layer]
#     defect_n1[cond_layer] = defect_indices[cond_layer]
#     defect_n2[cond_layer] = defect_indices[cond_layer]
#     defect_n3[cond_layer] = defect_indices[cond_layer]

#     defect_n0[~cond_layer] = defect_indices[~cond_layer] - 0.5
#     defect_n3[~cond_layer] = defect_indices[~cond_layer] + 0.5

#     temp = defect_indices[~cond_layer] - 0.5
#     temp[1::2] = temp[1::2] + 1
#     defect_n1[~cond_layer] = temp

#     temp = defect_indices[~cond_layer] - 0.5
#     temp[:-1:2] = temp[:-1:2] + 1
#     defect_n2[~cond_layer] = temp

#     index_list = np.arange(0, 4*defect_num, 4)
#     defect_n[index_list] = defect_n0
#     defect_n[index_list+1] = defect_n1
#     defect_n[index_list+2] = defect_n2
#     defect_n[index_list+3] = defect_n3
#     defect_n = np.unique(defect_n, axis=0)
#     defect_n = defect_n.astype(int)

#     return defect_n



# def plot_loop(
#             loop_coord, 
#             tube_radius=0.25, tube_opacity=0.5, tube_color=(0.5,0.5,0.5), if_add_head=True,
#             if_norm=False, 
#             norm_coord=[None,None,None], norm_color=(0,0,1), norm_length=20, 
#             norm_opacity=0.5, norm_width=1.0, norm_orient=1,
#             print_load_mayavi=False
#             ):

#     if print_load_mayavi == True:
#         now = time.time()
#         from mayavi import mlab
#         print(f'loading mayavi cost {round(time.time()-now, 2)}s')
#     else:
#         from mayavi import mlab

#     if if_add_head==True:
#         loop_coord = np.concatenate([loop_coord, [loop_coord[0]]])

#     mlab.plot3d(*(loop_coord.T), tube_radius=tube_radius, opacity=tube_opacity, color=tube_color)

#     if if_norm == True:
#         loop_N = get_plane(loop_coord) * norm_orient
#         loop_center = loop_coord.mean(axis=0)
#         for i, coord in enumerate(norm_coord):
#             if coord != None:
#                 loop_center[i] = coord
#         mlab.quiver3d(
#         *(loop_center), *(loop_N),
#         mode='arrow',
#         color=norm_color,
#         scale_factor=norm_length,
#         opacity=norm_opacity,
#         line_width=norm_width
#         ) 


# def plot_loop_from_n(
#                     n_box, 
#                     origin=[0,0,0], N=1, width=1, 
#                     tube_radius=0.25, tube_opacity=0.5, tube_color=(0.5,0.5,0.5), if_add_head=True,
#                     if_smooth=True, window_ratio=3, order=3, N_out=160,
#                     deform_funcs=[None,None,None],
#                     if_norm=False, 
#                     norm_coord=[None,None,None], norm_color=(0,0,1), norm_length=20, 
#                     norm_opacity=0.5, norm_width=1.0, norm_orient=1

#                     ):

#     loop_indices = defect_detect(n_box)
#     if len(loop_indices) > 0:
#         loop_indices = loop_indices + np.tile(origin, (np.shape(loop_indices)[0],1) )
#         loop_coord = sort_line_indices(loop_indices)/N*width
#         for i, func in enumerate(deform_funcs):
#             if func != None:
#                 loop_coord[:,i] = func(loop_coord[:,i])
#         if if_smooth == True:
#             loop_coord = smoothen_line(
#                                     loop_coord,
#                                     window_ratio=window_ratio, order=order, N_out=N_out,
#                                     mode='wrap'
#                                     )
#         plot_loop(
#                 loop_coord, 
#                 tube_radius=tube_radius, tube_opacity=tube_opacity, tube_color=tube_color,
#                 if_add_head=if_add_head,
#                 if_norm=if_norm,
#                 norm_coord=norm_coord, norm_color=norm_color, norm_length=norm_length, 
#                 norm_opacity=norm_opacity, norm_width=norm_width, norm_orient=norm_orient
#                     ) 


# def show_loop_plane(
#                     loop_box_indices, n_whole, 
#                     width=0, margin_ratio=0.6, upper=0, down=0, norm_index=0, 
#                     tube_radius=0.25, tube_opacity=0.5, scale_n=0.5, 
#                     if_smooth=True,
#                     print_load_mayavi=False
#                     ):
    
#     #! Unify origin (index or sapce unit)

#     # Visualize the disclination loop with directors lying on one cross section
    
#     if print_load_mayavi == True:
#         now = time.time()
#         from mayavi import mlab
#         print(f'loading mayavi cost {round(time.time()-now, 2)}s')
#     else:
#         from mayavi import mlab
#     from Nematics3D.field import select_subbox, local_box_diagonalize

#     def SLP_plot_plane(upper, down, d_box, grid, norm_vec, n_box, scale_n):

#         index = (d_box<upper) * (d_box>down)
#         index = np.where(index == True)
#         n_plane = n_box[index]
#         scalars = np.abs(np.einsum('ij, j -> i', n_plane, norm_vec))

#         X, Y, Z = grid
#         cord1 = X[index] - n_plane[:,0]/2
#         cord2 = Y[index] - n_plane[:,1]/2
#         cord3 = Z[index] - n_plane[:,2]/2

#         vector = mlab.quiver3d(
#                 cord1, cord2, cord3,
#                 n_plane[:,0], n_plane[:,1], n_plane[:,2],
#                 mode = '2ddash',
#                 scalars = scalars,
#                 scale_factor=scale_n,
#                 opacity = 1
#                 )
#         vector.glyph.color_mode = 'color_by_scalar'
#         lut_manager = mlab.colorbar(object=vector)
#         lut_manager.data_range=(0,1)

#     N = np.shape(n_whole)[0]
#     if width == 0:
#         width = N

#     # Find the region enclosing the loop. The size of the region is controlled by margin_ratio
#     sl0, sl1, sl2, _ = select_subbox(loop_box_indices, 
#                                 [N, N, N], 
#                                 margin_ratio=margin_ratio
#                                 )

#     # Select the local n around the loop
#     n_box = n_whole[sl0,sl1,sl2]

#     eigvec, eigval = local_box_diagonalize(n_box)

#     # The directors within one cross section of the loop will be shown
#     # Select the cross section by its norm vector
#     # The norm of the principle plane is the eigenvector corresponding to the smallest eigenvalue
#     norm_vec = eigvec[norm_index]

#     # Build the grid for visualization
#     x = np.arange( loop_box_indices[0][0], loop_box_indices[0][-1]+1 )/N*width
#     y = np.arange( loop_box_indices[1][0], loop_box_indices[1][-1]+1 )/N*width
#     z = np.arange( loop_box_indices[2][0], loop_box_indices[2][-1]+1 )/N*width
#     grid = np.meshgrid(x,y,z, indexing='ij')

#     # Find the height of the middle cross section: dmean
#     d_box = np.einsum('iabc, i -> abc', grid, norm_vec)
#     dmean = np.average(d_box)

#     down, upper = np.sort([down, upper])
#     if upper==down:
#         upper = dmean + 0.5
#         down  = dmean - 0.5
#     else:
#         upper = dmean + upper
#         down  = dmean + down

#     mlab.figure(bgcolor=(0,0,0))
#     SLP_plot_plane(upper, down, d_box, grid, norm_vec, n_box, scale_n)
#     plot_loop_from_n(
#                     n_box, 
#                     origin=loop_box_indices[:,0], N=N, width=width,
#                     tube_radius=tube_radius, tube_opacity=tube_opacity,
#                     if_smooth=if_smooth
#                     )

#     return dmean, eigvec, eigval

# # -----------------------------------------------------
# # Visualize the directors projected on principle planes
# # ----------------------------------------------------- 

# def show_plane_2Ddirector(
#                         n_box, height, 
#                         color_axis=(1,0), height_visual=0,
#                         space=3, line_width=2, line_density=1.5, 
#                         if_omega=True, S_box=0, S_threshold=0.18,
#                         if_cb=True, colormap='blue-red',
#                           ):
#     # Visualize the directors projected on principle planes

#     from mayavi import mlab
    
#     from plotdefect import get_streamlines
    
#     color_axis1 = color_axis / np.linalg.norm(color_axis) 
#     color_axis2 = np.cross( np.array([0,0,1]), np.concatenate( [color_axis1,[0]] ) )
#     color_axis2 = color_axis2[:-1]
    
#     x = np.arange(np.shape(n_box)[0])
#     y = np.arange(np.shape(n_box)[1])
#     z = np.arange(np.shape(n_box)[2])

#     indexy = np.arange(0, np.shape(n_box)[1], space)
#     indexz = np.arange(0, np.shape(n_box)[2], space)
#     iny, inz = np.meshgrid(indexy, indexz, indexing='ij')
#     ind = (iny, inz)

#     n_plot = n_box[height]

#     n_plane = np.array( [n_plot[:,:,1][ind], n_plot[:,:,2][ind] ] )
#     n_plane = n_plane / np.linalg.norm( n_plane, axis=-1, keepdims=True)

#     stl = get_streamlines(
#                 y[indexy], z[indexz], 
#                 n_plane[0].transpose(), n_plane[1].transpose(),
#                 density=line_density)
#     stl = np.array(stl)

#     connect_begin = np.where(np.abs( stl[1:,0] - stl[:-1,1]  ).sum(axis=-1) < 1e-5 )[0]
#     connections = np.zeros((len(connect_begin),2))
#     connections[:,0] = connect_begin
#     connections[:,1] = connect_begin + 1

#     lines_index = np.arange(np.shape(stl)[0])
#     disconnect = lines_index[~np.isin(lines_index, connect_begin)]

#     if height_visual == 0:
#         src_x = stl[:, 0, 0] * 0 + height
#     else:
#         src_x = stl[:, 0, 0] * 0 + height_visual
#     src_y = stl[:, 0, 0]
#     src_z = stl[:, 0, 1]

#     unit = stl[1:, 0] - stl[:-1, 0]
#     unit = unit / np.linalg.norm(unit, axis=-1, keepdims=True)

#     coe1 = np.einsum('ij, j -> i', unit, color_axis1)
#     coe2 = np.einsum('ij, j -> i', unit, color_axis2)
#     coe1 = np.concatenate([coe1, [coe1[-1]]])
#     coe2 = np.concatenate([coe2, [coe2[-1]]])
#     colors = np.arctan2(coe1,coe2)
#     nan_index = np.array(np.where(np.isnan(colors)==1))
#     colors[nan_index] = colors[nan_index-1]
#     colors[disconnect] = colors[disconnect-1]

#     src = mlab.pipeline.scalar_scatter(src_x, src_y, src_z, colors)
#     src.mlab_source.dataset.lines = connections
#     src.update()

#     lines = mlab.pipeline.stripper(src)
#     plot_lines = mlab.pipeline.surface(lines, line_width=line_width, colormap='blue-red')

#     if type(colormap) == np.ndarray:
#         lut = plot_lines.module_manager.scalar_lut_manager.lut.table.to_array()
#         lut[:, :3] = colormap
#         plot_lines.module_manager.scalar_lut_manager.lut.table = lut
    

#     if if_cb == True:
#         cb = mlab.colorbar(object=plot_lines, orientation='vertical', nb_labels=5, label_fmt='%.2f')
#         cb.data_range = (0,1)
#         cb.label_text_property.color = (0,0,0)
    

#     if if_omega == True: 

#         from scipy import ndimage

#         binimg = S_box<S_threshold
#         binimg[:height] = False
#         binimg[(height+1):] = False
#         binimg = ndimage.binary_dilation(binimg, iterations=1)
#         if np.sum(binimg) > 0:
#             labels, num_objects = ndimage.label(binimg, structure=np.zeros((3,3,3))+1)
#             if num_objects > 2:
#                 raise NameError('more than two parts')
#             elif num_objects == 1:
#                 print('only one part. Have to seperate points by hand')
#                 cross_coord = np.transpose(np.where(binimg==True))
#                 cross_center = np.mean(cross_coord, axis=0)
#                 cross_relative = cross_coord - np.tile(cross_center, (np.shape(cross_coord)[0],1))
#                 cross_dis = np.sum(cross_relative**2, axis=1)**(1/2)
#                 cross_to0 = cross_coord[cross_dis < np.percentile(cross_dis, 50), :]
#                 binimg[tuple(np.transpose(cross_to0))] = False
#             labels, num_objects = ndimage.label(binimg, structure=np.zeros((3,3,3))+1)
#             for i in range(1,3):  
#                 index = np.where(labels==i)
#                 X, Y, Z = np.meshgrid(x,y,z, indexing='ij')
#                 cord1 = X[index] - n_box[..., 0][index]/2
#                 cord2 = Y[index] - n_box[..., 1][index]/2
#                 cord3 = Z[index] - n_box[..., 2][index]/2
            
#                 pn = np.vstack((n_box[..., 0][index], n_box[..., 1][index], n_box[..., 2][index]))
#                 Omega = get_plane(pn.T)
#                 Omega= Omega * np.sign(Omega[0])
#                 scale_norm = 20
#                 if height_visual == 0:
#                     xvisual = cord1.mean()
#                 else:
#                     xvisual = height_visual
#                 mlab.quiver3d(
#                         xvisual, cord2.mean(), cord3.mean(),
#                         Omega[0], Omega[1], Omega[2],
#                         mode='arrow',
#                         color=(0,1,0),
#                         scale_factor=scale_norm,
#                         opacity=0.5
#                         )


# def show_loop_plane_2Ddirector(
#                                 n_box, S_box,
#                                 height_list, if_omega_list=[1,1,1], plane_list=[1,1,1],
#                                 height_visual_list=0, if_rescale_loop=True,
#                                 figsize=(1920, 1360), bgcolor=(1,1,1), camera_set=0,
#                                 if_norm=True, norm_length=20, norm_orient=1, norm_color=(0,0,1),
#                                 line_width=2, line_density=1.5,
#                                 tube_radius=0.75, tube_opacity=1,
#                                 print_load_mayavi=False, if_cb=True, n_colormap='blue-red'
#                                 ):
    
#     # Visualize the disclination loop with directors projected on several principle planes

#     if height_visual_list == 0:
#         height_visual_list = height_list
#         if_rescale_loop = False
#         parabola = None
#     if if_rescale_loop == True:
#         x, y, z = height_list
#         coe_matrix = np.array([
#                         [x**2, y**2, z**2],
#                         [x, y, z],
#                         [1,1,1]
#                         ])
#         del x, y, z
#         coe_parabola = np.dot(height_visual_list, np.linalg.inv(coe_matrix))
#         def parabola(x):
#             return coe_parabola[0]*x**2 + coe_parabola[1]*x + coe_parabola[2]
#     else:
#         def parabola(x):
#             return x


#     if print_load_mayavi == True:
#         now = time.time()
#         from mayavi import mlab
#         print(f'loading mayavi cost {round(time.time()-now, 2)}s')
#     else:
#         from mayavi import mlab

#     mlab.figure(size=figsize, bgcolor=bgcolor)

#     plot_loop_from_n(n_box, 
#                      tube_radius=tube_radius, tube_opacity=tube_opacity, 
#                      deform_funcs=[parabola,None,None],
#                      if_norm=if_norm,
#                      norm_coord=[height_visual_list[0],None,None], norm_length=norm_length, norm_orient=norm_orient, norm_color=norm_color,
#                      )

#     for i, if_plane in enumerate(plane_list):
#         if if_plane == True:
#             show_plane_2Ddirector(n_box, height_list[i], 
#                                   height_visual=height_visual_list[i], if_omega=if_omega_list[i], 
#                                   line_width=line_width, line_density=line_density,
#                                   S_box=S_box, if_cb=if_cb, colormap=n_colormap)

#     if camera_set != 0: 
#         mlab.view(*camera_set[:3], roll=camera_set[3])

        
# def plot_defect(n, 
#                 origin=[0,0,0], grid=128, width=200,
#                 if_plot_defect=True, defect_threshold=0, defect_color=(0.2,0.2,0.2), scale_defect=2,
#                 plot_n=True, n_interval=1, ratio_n_dist = 5/6,
#                 print_load_mayavi=False
#                 ):
    
#     # Visualize the disclinations within the simulation box

#     if plot_n == False and plot_defect == False:
#         print('no plot')
#         return 0
#     else:
#         if print_load_mayavi == True:
#             now = time.time()
#             from mayavi import mlab
#             print(f'loading mayavi cost {round(time.time()-now, 2)}s')
#         else:
#             from mayavi import mlab
#         mlab.figure(bgcolor=(1,1,1))

#     if plot_n == True:

#         nx = n[:,:,:,0]
#         ny = n[:,:,:,1]
#         nz = n[:,:,:,2]

#         N, M, L = np.shape(n)[:-1]

#         indexx = np.arange(0, N, n_interval)
#         indexy = np.arange(0, M, n_interval)
#         indexz = np.arange(0, L, n_interval)
#         ind = tuple(np.meshgrid(indexx, indexy, indexz, indexing='ij'))
        
#         x = indexx / grid * width + origin[0]
#         y = indexy / grid * width + origin[1]
#         z = indexz / grid * width + origin[2]
#         X, Y, Z = np.meshgrid(x,y,z, indexing='ij')

#         distance = n_interval / grid * width
#         n_length = distance * ratio_n_dist

#         coordx = X - nx[ind] * n_length / 2
#         coordy = Y - ny[ind] * n_length / 2
#         coordz = Z - nz[ind] * n_length / 2

#         phi = np.arccos(nx[ind])
#         theta = np.arctan2(nz[ind], ny[ind])

#         vector = mlab.quiver3d(
#                                 coordx, coordy, coordz,
#                                 nx[ind],ny[ind],nz[ind],
#                                 mode='cylinder',
#                                 scalars = (1-np.cos(2*phi))*(np.sin(theta%np.pi)+0.3),
#                                 scale_factor=n_length
#                                 )
        
#         vector.glyph.color_mode = 'color_by_scalar'
        
#         lut_manager = mlab.colorbar(object=vector)
#         lut_manager.data_range=(0,1.3)

#     if if_plot_defect == True:

#         defect_indices = defect_detect(n, threshold=defect_threshold)
#         defect_coords  = defect_indices / grid * width
#         defect_coords = defect_coords + origin
#         mlab.points3d(*(defect_coords.T), color=defect_color, scale_factor=scale_defect)









      

# def ordered_bulk_size(defect_indices, N, width, if_print=True):
#     '''
#     Compute the minimum distance from each point in a 3D grid the neirhboring defects.

#     Parameters
#     ----------
#     defect_indices : numpy.ndarray, shape (N,3)
#                      Array containing the coordinates of defect points. 
#                      M is the number of defect points.

#     N : int
#         Size of the cubic 3D grid in each dimension.

#     width : float
#             Width of the simulation box in the unit of real length (not indices).

#     if_print : bool, optional
#               Flag to print the time taken for each octant.
#               Default is True.

#     Returns
#     -------
#     dist_min : numpy.ndarray, shape (N^3,)
#                Array containing the minimum distances from each point in the 3D grid to the nearest defect.

#     Dependencies
#     ------------
#     - numpy: 1.22.0
#     - scipy: 1.7.3
#     '''
#     from itertools import product
#     import time

#     from scipy.spatial.distance import cdist

#     # Ensure defect indices are within the periodic boundary conditions
#     defect_indices = defect_indices % N

#     # Generate the coordinates of each point in the 3D grid.
#     grid = np.array(list(product(np.arange(N), np.arange(N), np.arange(N))))

#     # Divide defect and grid points into octants based on their positions in each dimension
#     defect_check0 = defect_indices[:,0] < int(N/2)
#     defect_check1 = defect_indices[:,1] < int(N/2)
#     defect_check2 = defect_indices[:,2] < int(N/2)

#     defect_box = [defect_indices[ np.where(  defect_check0 *  defect_check1 *  defect_check2 ) ],
#                   defect_indices[ np.where( ~defect_check0 *  defect_check1 *  defect_check2 ) ],
#                   defect_indices[ np.where(  defect_check0 * ~defect_check1 *  defect_check2 ) ],
#                   defect_indices[ np.where(  defect_check0 *  defect_check1 * ~defect_check2 ) ],
#                   defect_indices[ np.where( ~defect_check0 * ~defect_check1 *  defect_check2 ) ],
#                   defect_indices[ np.where( ~defect_check0 *  defect_check1 * ~defect_check2 ) ],
#                   defect_indices[ np.where(  defect_check0 * ~defect_check1 * ~defect_check2 ) ],
#                   defect_indices[ np.where( ~defect_check0 * ~defect_check1 * ~defect_check2 ) ]]

#     grid_check0 = grid[:,0] < int(N/2)
#     grid_check1 = grid[:,1] < int(N/2)
#     grid_check2 = grid[:,2] < int(N/2)

#     grid_box = []
#     grid_box = [grid[ np.where(  grid_check0 *  grid_check1 *  grid_check2 ) ],
#                 grid[ np.where( ~grid_check0 *  grid_check1 *  grid_check2 ) ],
#                 grid[ np.where(  grid_check0 * ~grid_check1 *  grid_check2 ) ],
#                 grid[ np.where(  grid_check0 *  grid_check1 * ~grid_check2 ) ],
#                 grid[ np.where( ~grid_check0 * ~grid_check1 *  grid_check2 ) ],
#                 grid[ np.where( ~grid_check0 *  grid_check1 * ~grid_check2 ) ],
#                 grid[ np.where(  grid_check0 * ~grid_check1 * ~grid_check2 ) ],
#                 grid[ np.where( ~grid_check0 * ~grid_check1 * ~grid_check2 ) ]]

#     # To same memory, consider half of grid points in each octant for each calculation
#     size = len(grid_box[0])
#     half = int(len(grid_box[0])/2)

#     dist_min = np.zeros(N**3)

#     # Calculate!
#     for i, box in enumerate(grid_box):
#         start = time.time()
#         dist_min[ i*size:i*size+half ] = np.min(cdist(box[:half], defect_box[i]), axis=-1)
#         dist_min[ i*size+half:(i+1)*size ] = np.min(cdist(box[half:], defect_box[i]), axis=-1)
#         print(time.time()-start)

#     # Change the unit of distances into real length,    
#     dist_min = dist_min / N * width

#     return dist_min



# def save_xyz(fname, atoms, lattice=None):
#     """
#     Saves data to a file in extended XYZ format. This function will attempt to
#     produce an extended XYZ file even if the input data is incomplete.
#     Usually used to store information of disclination loops.
#     Created by Matthew E. Peterson.

#     Parameters
#     ----------
#     fname : str or pathlib.Path
#             File to write data to.

#     atoms : pandas.DataFrame
#             Atomic data to be saved.

#     lattice : array_like (default None)
#               Lattice vectors of the system, so that lattice[0] is the first lattice vector.

#     Raises
#     ------
#     KeyError
#         If a key in `cols` does not appear in `data`.
#     """

#     import re
#     import os
#     import gzip
#     import shutil

#     gzipped = fname.endswith('.gz')
#     if gzipped:
#         fname = fname[:-3]

#     def map_type(dtype):
#         if np.issubdtype(dtype, np.integer):
#             return 'I'
#         elif np.issubdtype(dtype, np.floating):
#             return 'R'
#         else:
#             return 'S'

#     def collapse_names(names):
#         cols = dict()
#         for name in names:
#             col = re.sub(r'-\w+$', '', name)
#             try:
#                 cols[col].append(name)
#             except:
#                 cols[col] = [name]
#         return cols

#     with open(fname, 'w') as f:
#         # the first line is simply the number of atoms
#         f.write(f"{len(atoms)}\n")

#         # write lattice if given
#         header = ""
#         if lattice is not None:
#             lattice = np.asarray(lattice)
#             lattice = ' '.join(str(x) for x in lattice.ravel())
#             header += f'Lattice="{lattice}" '

#         # now determine property line
#         cols = collapse_names(atoms.columns)
#         props=[]
#         for col, names in cols.items():
#             name = names[0] 
#             tp=map_type(atoms.dtypes[name])
#             dim=len(names)
#             props.append(f"{col}:{tp}:{dim}")
#         props = ':'.join(props)
#         header += f'Properties={props}\n'

#         f.write(header)

#         # writing the atomic data is easy provided we are using a DataFrame
#         atoms.to_csv(f, header=None, sep=" ", float_format="%f", index=False)

#     if gzipped:
#         with open(fname, 'rb') as f_in:
#             with gzip.open(f"{fname}.gz", 'wb') as f_out:
#                 shutil.copyfileobj(f_in, f_out)

#         os.remove(fname)


# def extract_loop(defect_indices, N, dilate_times=2,
#                  save_path='.', save_head=''):
    
#     '''
#     Given the defect indices in 3D grids, extract loops and calculate their number of holes.
#     Using binary dilation and Euler number calculation. 
#     Saves loop information and coordinates in files.

#     Parameters
#     ----------
#     defect_indices : array_like, (M, 3)
#                      Array containing the indices of defects in the 3D grid.
#                      M is the number of defects, and each row contains (x, y, z) indices.

#     N : int
#         Size of the 3D grid along each dimension.

#     dilate_times : int, optional
#                    Number of iterations for binary dilation to thicken the disclination lines.
#                    Default is 2.

#     save_path : str, optional
#                 Path to save loop information files. 
#                 Default is current directory.

#     save_head : str, optional
#                 Prefix for saved files. 
#                 Default is an empty string.

#     Saves
#     -----
#     'summary.xyz.gz' : summary, pandas file
#                        The information of each point of thickened disclination loops.
#                        Include x, y, z coordinates (indices), genus and label of this loop.

#     'summary_wrap.xyz.gz' : summary_wrap, pandas file
#                             Same with summary except the coordinates are wrapped in periodic boundary condition

#     '{lable}box_{g}.txt' : box, array-like, (3,2)
#                            The vortices of box containing one thickened disclination loop.
#                            g = genus of this loop

#     '{lable}index_{g}.txt' : coord, array-like, (M,3)
#                              The indices of each point in one thickened disclination loop.
#                              g = genus of this loop, M is the number of points in the loop.

#     'grid_g.npy' : grid_result, array-like, (N,N,N)
#                    The defects indices and genus represented by grid of box.
#                    N is the size of the 3D grid along each dimension.
#                    If one point is 0, there is no defect here.
#                    If one point is x (not zero), there is a defect here and the genus of this line is x-1.
    
#     Dependencies
#     ------------
#     - numpy : 1.22.0
#     - scipy : 1,7.3
#     - skimage : 0.19.1
#     - pandas : 2.0.2
#     '''


#     from scipy import ndimage
#     from skimage.measure import euler_number
#     import pandas as pd

#     # Ensure save_path ends with '/'
#     save_path = save_path + '/'

#     # Wrap the defect
#     defect_indices = (defect_indices%N).astype(int)

#     # Create a binary grid representing the defect indices
#     grid_origin = np.zeros((N, N, N))
#     grid_origin[tuple(defect_indices.T)] = 1

#     # Expand the defect grid to handle periodic boundary conditions
#     grid = np.zeros( (2*N, 2*N, 2*N ) )
#     grid[:N, :N, :N]            = grid_origin
#     grid[N:2*N, :N, :N]         = grid_origin
#     grid[:N, N:2*N, :N]         = grid_origin
#     grid[:N, :N, N:2*N]         = grid_origin
#     grid[N:2*N, N:2*N, :N]      = grid_origin
#     grid[:N, N:2*N, N:2*N]      = grid_origin
#     grid[N:2*N, :N, N:2*N]      = grid_origin
#     grid[N:2*N, N:2*N, N:2*N]   = grid_origin

#     # Perform binary dilation on the extended grid to thicken the lines
#     binimg = ndimage.binary_dilation(grid, iterations=dilate_times)

#     # Count the number of low-order points
#     num_pts = int(np.count_nonzero(binimg)/8)
#     print(f"Found {num_pts} low-order points")

#     # Prepare the array to record genus of each defect
#     grid_origin = grid_origin/2
#     grid_result = np.zeros((N, N, N))

#     # Initialize arrays to store loop information
#     summary = np.empty(8*num_pts,
#                        dtype=[
#                            ("pos-x", "uint16"),
#                            ("pos-y", "uint16"),
#                            ("pos-z", "uint16"),
#                            ("genus", "int16"),
#                            ("label", "uint16"),
#                        ]
#                        )
    
#     summary_wrap = np.empty(8*num_pts,
#                        dtype=[
#                            ("pos-x", "uint16"),
#                            ("pos-y", "uint16"),
#                            ("pos-z", "uint16"),
#                            ("genus", "int16"),
#                            ("label", "uint16"),
#                        ]
#                        )
    
#     # Label connected components in the binary grid
#     labels, num_objects = ndimage.label(binimg)

#     offset = 0
#     label = 0
#     loop = 0
#     for i, obj in enumerate(ndimage.find_objects(labels)):
        
#         # Ensure object boundaries are within the extended grid
#         xlo = max(obj[0].start, 0)
#         ylo = max(obj[1].start, 0)
#         zlo = max(obj[2].start, 0)
#         xhi = min(obj[0].stop, 2*N)
#         yhi = min(obj[1].stop, 2*N)
#         zhi = min(obj[2].stop, 2*N)
#         boundary = [xlo, ylo, zlo, xhi, yhi, zhi]
        
#         # Exclude the crosses and loops outside of the box and  
#         if (0 in boundary) or (2*N in boundary) or max(xlo, ylo, zlo)>N:
#             continue
        
#         label += 1
        
#         # Define the object within the extended grid
#         obj = (slice(xlo, xhi), slice(ylo, yhi), slice(zlo, zhi))
        
#         # Extract the object from the labeled grid
#         img = (labels[obj] == i+1)
    
#         # calculate Euler number, defined as # objects + # holes - # loops
#         # we do not have holes
#         g = euler_number(img, connectivity=1)

#         # calculate the number of loop by Euler number
#         g = 1 - g
        
#         # box : The vortices of box containing the loop
#         # coord : The indices of each point of thickened loop
#         pos = np.nonzero(img)
#         shift = pos[0].size
        
#         box = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]], dtype=int)
#         coord = np.array([pos[0] + xlo, pos[1] + ylo, pos[2] + zlo])

#         np.savetxt(save_path + save_head + f"{label}box_{g}.txt", box, fmt="%d")
#         np.savetxt(save_path + save_head + f"{label}index_{g}.txt", coord, fmt="%d")

#         if g == 1:
#             loop += 1

#         # Update summary arrays with loop information
#         summary['pos-x'][offset:offset+shift] = coord[0]
#         summary['pos-y'][offset:offset+shift] = coord[1]
#         summary['pos-z'][offset:offset+shift] = coord[2]
#         summary['genus'][offset:offset+shift] = g
#         summary['label'][offset:offset+shift] = label
        
#         summary_wrap['pos-x'][offset:offset+shift] = coord[0]%N
#         summary_wrap['pos-y'][offset:offset+shift] = coord[1]%N
#         summary_wrap['pos-z'][offset:offset+shift] = coord[2]%N
#         summary_wrap['genus'][offset:offset+shift] = g
#         summary_wrap['label'][offset:offset+shift] = label
    
#         offset += shift

#         # Update the genus information of each defect
#         grid_result[tuple(coord%N)] = g+1
        
#     # grid_result : each point of defect labeled by the genus of lines that the defect belongs to.
#     grid_result = grid_result + grid_origin
#     grid_result[grid_result%1==0] = 0.5
#     grid_result = grid_result - 0.5

#     np.save(save_path + save_head + "grid_g.npy", grid_result)
    
#     # Trim summary arrays to remove unused space
#     summary = summary[:offset]
#     summary_wrap = summary_wrap[:offset]
#     summary = pd.DataFrame(summary)
#     summary_wrap = pd.DataFrame(summary_wrap)

#     save_xyz(save_path + save_head + "summary.xyz.gz", summary)
#     save_xyz(save_path + save_head + "summary_wrap.xyz.gz", summary_wrap)
                    
#     print(f'Found {loop} loops\n')

# def defect_find_vicinity_Q(defect_indices, n, S=0, num_add=3, is_boundary_periodic=0):

#     from .field import interpolateQ

#     defect_vicinity_grid = defect_vicinity_grid_local(defect_indices, num_add=num_add)
#     defect_vicinity_Q = interpolateQ(n, defect_vicinity_grid, S=S, is_boundary_periodic=is_boundary_periodic)

#     return defect_vicinity_Q


# def defect_vicinity_rotation(defect_indices, n, S=0, num_add=3, is_boundary_periodic=0):

#     from .field import diagonalizeQ
#     from .general import get_plane

#     defect_vicinity_Q = defect_find_vicinity_Q(defect_indices, n, 
#                                                S=S, num_add=num_add, is_boundary_periodic=is_boundary_periodic)
#     defect_vicinity_n = diagonalizeQ(defect_vicinity_Q)[1]

#     # same_orient = np.sign(np.einsum( 'abi, abi -> ab', defect_vicinity_n[:, 1:], defect_vicinity_n[:, :-1] ))
#     # same_orient = np.cumprod( same_orient, axis=-1)[:,-1]
#     # final_product = np.einsum( 'a, ai, ai -> a', same_orient, defect_vicinity_n[:,0], defect_vicinity_n[:,-1] )

#     norm = get_plane(defect_vicinity_n)

#     return norm

# def defect_vicinity_grid_local(defect_indices, num_add=3):

#     num = num_add + 2
#     length = 4 * num - 4

#     result = np.zeros( (np.shape(defect_indices)[0], length, 3) )

#     indexx = np.isclose(defect_indices[:, 0], np.round(defect_indices[:, 0]))
#     indexy = np.isclose(defect_indices[:, 1], np.round(defect_indices[:, 1]))
#     indexz = np.isclose(defect_indices[:, 2], np.round(defect_indices[:, 2]))

#     defectx = defect_indices[indexx]
#     defecty = defect_indices[indexy]
#     defectz = defect_indices[indexz]

#     squarex = get_square(1, num, dim=3)
#     squarey = squarex.copy()
#     squarey[:, [0, 1]] = squarey[:, [1, 0]]
#     squarez = squarex.copy()
#     squarez[:, [0, 1]] = squarez[:, [1, 0]]
#     squarez[:, [1, 2]] = squarez[:, [2, 1]]

#     defectx = defectx - np.broadcast_to([0.0, 0.5, 0.5],(np.shape(defectx)[0],3))
#     defecty = defecty - np.broadcast_to([0.5, 0.0, 0.5],(np.shape(defecty)[0],3))
#     defectz = defectz - np.broadcast_to([0.5, 0.5, 0.0],(np.shape(defectz)[0],3))

#     defectx = np.repeat(defectx, length, axis=0).reshape(np.shape(defectx)[0],length,3)
#     defecty = np.repeat(defecty, length, axis=0).reshape(np.shape(defecty)[0],length,3)
#     defectz = np.repeat(defectz, length, axis=0).reshape(np.shape(defectz)[0],length,3)

#     defectx =  defectx + np.broadcast_to(squarex, (np.shape(defectx)[0], length,3))
#     defecty =  defecty + np.broadcast_to(squarey, (np.shape(defecty)[0], length,3))
#     defectz =  defectz + np.broadcast_to(squarez, (np.shape(defectz)[0], length,3))

#     result[indexx] = defectx
#     result[indexy] = defecty
#     result[indexz] = defectz

#     return result

# def defect_detect_precise(n, S=0, defect_indices=0,
#                           threshold1=0.5, threshold2=-0.9, is_boundary_periodic=0, planes=[1,1,1], num_add=3
#                           ):

#     if isinstance(defect_indices, int):
#         defect_indices = defect_detect(n, threshold=threshold1, is_boundary_periodic=is_boundary_periodic, planes=planes)

#     final_product = defect_vicinity_analysis(defect_indices, n, S=S, num_add=num_add, is_boundary_periodic=is_boundary_periodic)[0]

#     return defect_indices[final_product<threshold2]


# def visualize_nematics_field_old(n=[0], S=[0],
#                              plotn=True, plotS=True, plotdefects=False,
#                              space_index_ratio=1, sub_space=1, origin=(0,0,0), expand_ratio=1,
#                              n_opacity=1, n_interval=(1,1,1), n_shape='cylinder', n_width=2, n_plane_index=[],
#                              n_color_func=n_color_func_default, n_length_ratio_to_dist=0.8,
#                              n_if_colorbar=True, n_colorbar_params={}, n_colorbar_range='default',
#                              S_threshold=0.45, S_opacity=1, S_plot_params={},
#                              S_if_colorbar=True, S_colorbar_params={}, S_colorbar_range=(0,1),
#                              defect_opacity=0.8, defect_size_ratio_to_dist=2, defect_color=(0,0,0),
#                              is_boundary_periodic=False, defect_threshold=0, defect_print_time=False,
#                              new_figure=True, bgcolor=(1,1,1), fgcolor=(0,0,0), if_axes=False,
#                              ):

#     #! float n_interval with interpolation
#     #! different styles for different planes
#     #! input defect indices
#     #! plotdefects smooth
#     #! axes without number
#     #! with class
#     #! warning about boundary condition of classfication of defect lines
#     #!

#     """
#     Visualize a 3D nematics field using Mayavi. 
#     The function provides options to plot the director field, scalar order parameter contours, and defect lines.
#     The locations of defect points are automatically analyzed by the input director field.
    
#     Parameters
#     ----------
#     n : numpy array, N x M x L x 3, optional
#         The director of each grid point.
#         Default is [0], indicating no data of n.
#         The shape of n, (N, M, L), must match the shape of S if S is plotted with n or defects.

#     S : numpy array, N x M x L, optional
#         the scalar order parameter of each grid point.
#         Default is [0], which means no data of S.
#         The shape of n, (N, M, L), must match the shape of S if S is plotted with n or defects.

#     plotn : bool, optional
#             If True, plot the director field. 
#             Default is True.   

#     plotS : bool, optional
#             If True, plot contour of scalar order parameter. 
#             Default is True.

#     plotdefects : bool, optional
#                   If True, plot the defect points calculated by the director field. 
#                   Default is False.

#     space_index_ratio : float or array of three floats, optional
#                         Ratio between the unit of real space to the unit of grid indices.
#                         If the box size is N x M x L and the size of grid of n and S is n x m x l,
#                         then space_index_ratio should be (N/n, M/m, L/l).
#                         If a single float x is provided, it is interpreted as (x, x, x).
#                         Default is 1.

#     sub_space : int or array of int as ((xmin, xmax), (ymin, ymax), (zmin, zmax)), optional
#                 Specifies a sub-box to visualize.
#                 If an array, it provides the range of sub box by the indices of grid
#                 If an int, it represents a sub box in the center of the whole box, where the zoom rate is sub_space.
#                 Default is 1, as the whole box is visualized.

#      origin : array of three floats, optional
#               Origin of the plot, translating the whole system in real space
#               Default is (0, 0, 0), as the system is not translated 

#     expand_ratio : float or array of three floats, optional
#                    Ratio to stretch or compress the plot along each dimension.
#                    Usually used to see some structure more clearly.
#                    If a single float x is provided, it is interpreted as (x, x, x).
#                    Default is 1, as no stretch or compress.  
#                    Warning: if expand_ratio is not 1 or (1,1,1), the plot will be deformed, so that it does NOT stand for the real space anymore.


#     n_opacity : float, optional
#                 Opacity of the director field. 
#                 Default is 1.

#     n_interval : int or array of three int, optional
#                  Interval between directors in each dimension, specified in grid indices.
#                  For example, if n_interval = (1,2,2), it will plot all directors in x-dimension but plot directors alternatively in y and z-dimension.
#                  So large n_internval leads to dilute directors in the plot.
#                  If a single int x is provided, it is interpreted as (x, x, x).
#                  Default is (1, 1, 1), plotting all directors.

#     n_shape : str, optional
#               Shape of the director glyphs. It's usually 'cylinder' or '2ddash'.
#               Default is 'cylinder'.
#               Warning: some of glyph shapes can not be controlled by n_width. 
#               For example, n_width works for n_shape='2ddash' but does not works for n_shape='cylinder'.

#     n_width : float, optional
#               Width of the director glyphs. 
#               It does not work for all shapes. For example, it works for n_shape='2ddash' but does not work for n_shape='cylinder'.
#               Default is 2.

#     n_color_func : function or array of three float, optional
#                    Color for the director glyphs. 
#                    If it is a function, the input should be an array of shape (N, M, L, 3) and output should be array of shape (N,M,L), which is the scalars to color the directors.
#                    If it is an array, it specifies RGB color for all directors.
#                    For example, if n_color_func=(1,0,0), all directors will be red.
#                    Default is n_color_func_default, a function I personally used to distinguish directors with different orientation.
#                    Warning: Two directors with the SAME color could have DIFFERENT orientation, 
#                    while two directors with different color always have different orientations.
                        
#     n_length_ratio_to_dist : float, optional
#                              Ratio bwtween the length of director's glyph and the minimal distance between two glyphs.
#                              Used to set the length of direcor's glyph.
#                              Default is 0.8.

#     n_if_colorbar : bool, optional
#                     If True, display a colorbar for the director field when n_color_func is a function.
#                     Default is True.

#     n_colorbar_params : dict, optional
#                         Parameters for the director field colorbar. 
#                         For example, {"orientation": "vertical"} will make the colorbar vertical.
#                         Check the document of mlab.colorbar() to see all options.
#                         Default is {}.

#     n_colorbar_range : array of two float, or "default', or 'max', optional
#                        Range for the director field colorbar. 
#                        If n_colorbar_range is an array of two floats (a,b), the range will be (a,b).
#                        If n_colorbar_range='max', the range will cover the minimum and maximum value of scalars (read the introduction of n_color_func).
#                        If n_colorbar_range='default', it only works for n_color_func=n_color_func_default. The range will be (0, 2.6) as the therotical range of the default function.
#                        Default is 'default'.

#     S_threshold : float, optional
#                   Threshold for plotting scalar order parameter contours. 
#                   Default is 0.45.

#     S_opacity : float, optional
#                 Opacity of the scalar order parameter field. Default is 1.

#     S_plot_params : dict, optional
#                     Parameters for plotting the scalar order parameter field, except opacity and threshold.
#                     For example, {'color': (1,1,1)} will make the contours white.
#                     Check the document of mlab.contour3D() to see all options.
#                     Default is {}.

#     S_if_colorbar : bool, optional
#                     If True, display a colorbar for the scalar order parameter field based on the value of S. 
#                     If the color of contours is set, colorbar will NOT be presented.
#                     Default is True.

#     S_colorbar_params : dict, optional
#                         Parameters for the scalar order parameter colorbar. 
#                         For example, {"orientation": "vertical"} will make the colorbar vertical.
#                         Check the document of mlab.colorbar() to see all options.
#                         Default is {}.

#     S_colorbar_range : array of two float, optional
#                        Range for the scalar order parameter colorbar. 
#                        Default is (0, 1).

#     defect_opacity : float, optional
#                      Opacity of the defect points. 
#                      Default is 0.8.

#     defect_size_ratio_to_dist : float, optional
#                                 The ratio bwtween the size of defect points to the minimal distance between two glyphs.
#                                 Used to set the size of defect points.
#                                 Default is 2.

#     is_boundary_periodic : bool, optional
#                         If True, use periodic boundary conditions during defects detection. 
#                         Default is False.
#                         Warning: If only part of the box is selected by sub_space so that it will not have the periodic boundary condition,
#                         is_boundary_periodic will be reset to False automatically if it is initially True.

#     defect_threshold : float, optional
#                        Threshold for detecting defects. 
#                        When calculating the winding number, if the inner product of neighboring directors after one loop is smaller than defect_threshold,
#                        the center of the loop is identified as one defect.
#                        Default is 0.

#     defect_print_time : bool, optional
#                         If True, print the time taken to detect defects in each dimension. 
#                         Default is False.

#     new_figure : bool, optional
#                  If True, create a new figure for the plot. 
#                  Default is True.

#     bgcolor : array of three floats, optional
#               Background color of the plot in RGB.
#               Default is (1,1,1), white.

#     fgcolor : array of three floats, optional
#               Foreground color of the plot in RBG.
#               It is the color of all text annotation labels (axes, orientation axes, scalar bar labels). 
#               It should be sufficiently far from bgcolor to see the annotation texts.
#               Default is (0,0,0), black

#     if_axes : bool, optional
#               If True, display axes in the plot in the unit of real space. 
#               Default is False.

#     Returns
#     -------
#     S : numpy array, N x M x L
#         the biggest eigenvalue as the scalar order parameter of each grid

#     n : numpy array, N x M x L x 3
#         the eigenvector corresponding to the biggest eigenvalue, as the director, of each grid.


#     Dependencies
#     ------------
#     - NumPy: 1.26.4
#     - mayavi: 4.8.2

#     """

#     # examine the input data
#     n = np.array(n)
#     S = np.array(S)
#     if S.size==1 and plotS==True:
#         raise NameError("No data of S input to plot contour plot of S.")
#     if n.size==1 and (plotn==True or plotdefects==True):
#         raise NameError("no data of n input to plot director field or defects.")
#     if S.size!=1 and n.size!=1:
#         if np.linalg.norm( np.array(np.shape(n)[:3]) - np.array(np.shape(S)) ) != 0:
#             raise NameError('The shape of n and S should be the same.')

#     from mayavi import mlab

#     # examine the input parameters
#     if len(np.shape([space_index_ratio])) == 1:
#         space_index_ratio = (space_index_ratio, space_index_ratio, space_index_ratio)
#     if len(np.shape([n_interval])) == 1:
#         n_interval = (n_interval, n_interval, n_interval)
#     if len(np.shape([expand_ratio])) == 1:
#         expand_ratio = np.array([expand_ratio, expand_ratio, expand_ratio])
#     else:
#         expand_ratio = np.array(expand_ratio)
#     if np.linalg.norm(expand_ratio-1) != 0 and if_axes==True:
#         print("\nWarning: The expand_ratio is not (1,1,1)")
#         print("This means the distance between glyphs are changed")
#         print("The values on the coordinate system (axes of figure) does NOT present the real space.")
#     if n_shape=='cylinder' and n_width!=2:
#         print('\nWarning: n_shape=cylinder, whose thickness can not be controlled by n_width.')


#     # the basic axes for the plotting in real space of the entire box
#     if np.size(n) > 1:
#         Nx, Ny, Nz = np.array(np.shape(n)[:3]) 
#     else:
#         Nx, Ny, Nz = np.array(np.shape(S)[:3])    
#     x = np.linspace(0, Nx*space_index_ratio[0]-1, Nx)
#     y = np.linspace(0, Ny*space_index_ratio[1]-1, Ny)
#     z = np.linspace(0, Nz*space_index_ratio[2]-1, Nz)
#     X, Y, Z = np.meshgrid(x,y,z, indexing='ij')

#     # create a new figure with the given background if needed
#     if new_figure:
#         mlab.figure(bgcolor=bgcolor, fgcolor=fgcolor)

#     # select the sub box to be plotted
#     if len(np.shape([sub_space])) == 1:
#         indexx = np.arange( int(Nx/2 - Nx/2/sub_space), int(Nx/2 + Nx/2/sub_space) )
#         indexy = np.arange( int(Ny/2 - Ny/2/sub_space), int(Ny/2 + Ny/2/sub_space) )
#         indexz = np.arange( int(Nz/2 - Nz/2/sub_space), int(Nz/2 + Nz/2/sub_space) )
#     else:
#         indexx = np.arange( sub_space[0][0], sub_space[0][-1]+1 )
#         indexy = np.arange( sub_space[1][0], sub_space[1][-1]+1 )
#         indexz = np.arange( sub_space[2][0], sub_space[2][-1]+1 )

#     inx, iny, inz = np.meshgrid(indexx, indexy, indexz, indexing='ij')
#     ind = (inx, iny, inz)

#     #plot defects
#     if plotdefects:

#         if sub_space != 1 and is_boundary_periodic == True:
#             print('\nWarning: The periodic boundary condition is only possible for the whole box')
#             print('Automatically deactivate the periodic boundary condition')
#             print('Read the document for more information.')
#             is_boundary_periodic = False

#         # find defects
#         from .disclination import defect_detect
#         print('\nDetecting defects')
#         defect_indices = defect_detect(n[ind], is_boundary_periodic=is_boundary_periodic, threshold=defect_threshold,
#                                        print_time=defect_print_time)
#         print('Finished!')

#         # Move the defect points to have the correct location in real space
#         defect_indices = np.einsum('na, a, a -> na',
#                                    defect_indices, space_index_ratio, expand_ratio)

#         if len(np.shape([sub_space])) == 1:
#             start_point = [x[int(Nx/2 - Nx/2/sub_space)],
#                            y[int(Ny/2 - Ny/2/sub_space)],
#                            z[int(Nz/2 - Nz/2/sub_space)]]
#         else:
#             start_point = [x[sub_space[0][0]],
#                            y[sub_space[1][0]],
#                            z[sub_space[2][0]]]

#         defect_indices = defect_indices + np.broadcast_to(start_point, (np.shape(defect_indices)[0],3))
#         defect_indices = defect_indices + np.broadcast_to(origin, (np.shape(defect_indices)[0],3))

#         # set the size of defect points
#         distx = (x[indexx[1]] - x[indexx[0]]) * expand_ratio[0]
#         disty = (y[indexy[1]] - y[indexy[0]]) * expand_ratio[1]
#         distz = (z[indexz[1]] - z[indexz[0]]) * expand_ratio[2]
#         dist = [distx, disty, distz]
#         defect_size = np.min(dist) * defect_size_ratio_to_dist
#         print(f'\nSize of defect points: {defect_size}')

#         # make plot of defect points if there is no need to plot the smoothened lines
#         mlab.points3d(
#             defect_indices[:,0], defect_indices[:,1], defect_indices[:,2],
#             scale_factor=defect_size, opacity=defect_opacity, color=defect_color
#             )





#     # plot contours of S
#     if plotS:

#         if np.min(S[ind])>=S_threshold or np.max(S[ind])<=S_threshold:

#             print(f'\nThe range of S is ({np.min(S[ind])}, {np.max(S[ind])})')
#             print('The threshold of contour plot of S is out of range')
#             print('No S region is plotted')

#         else:
#             # the grid in real space to plot S
#             cord1 = X[ind]*expand_ratio[0] + origin[0]
#             cord2 = Y[ind]*expand_ratio[1] + origin[1]
#             cord3 = Z[ind]*expand_ratio[2] + origin[2]

#             # check if the color of contours is set
#             S_plot_color = S_plot_params.get('color')

#             # make plot
#             S_region = mlab.contour3d(cord1, cord2, cord3, S[ind], 
#                                       contours=[S_threshold], opacity=S_opacity, **S_plot_params)
#             # make colorbar of S
#             if S_plot_color == None:
#                 if S_if_colorbar:
#                     S_colorbar_label_fmt    = S_colorbar_params.get('label_fmt', '%.2f')
#                     S_colorbar_nb_labels    = S_colorbar_params.get('nb_labels', 5)
#                     S_colorbar_orientation  = S_colorbar_params.get('orientation', 'vertical')
#                     S_lut_manager = mlab.colorbar(object=S_region,
#                                                   label_fmt=S_colorbar_label_fmt, 
#                                                   nb_labels=S_colorbar_nb_labels, 
#                                                   orientation=S_colorbar_orientation,
#                                                   **S_colorbar_params)
#                     S_lut_manager.data_range=(S_colorbar_range)
#                     if plotn and n_if_colorbar:
#                         print('\nWarning2: The colorbars for n and S both exist.')
#                         print('These two colorbars might interfere with each other visually.')
#             elif S_if_colorbar==True:
#                 print('\nThe color of S is set. So there is no colorbar for S.')

#     # plot directors
#     if plotn:

#         # the axes to plot n in real space is selected by n_interval
#         indexx_n = indexx[::n_interval[0]]
#         indexy_n = indexy[::n_interval[1]]
#         indexz_n = indexz[::n_interval[2]]
#         indexall = (indexx, indexy, indexz)

#         inx, iny, inz = np.meshgrid(indexx_n, indexy_n, indexz_n, indexing='ij')
#         ind = (inx, iny, inz)

#         # set the length of directors' glyph
#         distx = (x[indexx_n[1]] - x[indexx_n[0]]) * expand_ratio[0]
#         disty = (y[indexy_n[1]] - y[indexy_n[0]]) * expand_ratio[1]
#         distz = (z[indexz_n[1]] - z[indexz_n[0]]) * expand_ratio[2]
#         dist = [distx, disty, distz]
#         n_length = np.min(dist) * n_length_ratio_to_dist
#         print(f'\nThe length of directors\' glyph: {n_length}')

#         # examine if directors are colored by function or uniform color
#         try:
#             len(n_color_func)
#             scalars = X[ind]*0
#             n_color_scalars = False
#             n_color = n_color_func
#         except:
#             scalars = n_color_func(n[ind])
#             n_color_scalars = True
#             n_color = (1,1,1)

#         # if the planes of n are not set, use the default planes selected by n_interval
#         if len(n_plane_index) == 0:
            
#             # move the location of glyph so that the middle points of all glyphs form lattice
#             # this comes from the fact and directors are lines without directions
#             cord1 = (X[ind])*expand_ratio[0] - n[ind][..., 0]*n_length/2 + origin[0]
#             cord2 = (Y[ind])*expand_ratio[1] - n[ind][..., 1]*n_length/2 + origin[1]
#             cord3 = (Z[ind])*expand_ratio[2] - n[ind][..., 2]*n_length/2 + origin[2]

#             nx = n[ind][..., 0]
#             ny = n[ind][..., 1]
#             nz = n[ind][..., 2]
            
#         else:
#             cord1 = []
#             cord2 = []
#             cord3 = []
#             nx = []
#             ny = []
#             nz = []
#             scalars = []
#             for i, planes in enumerate(n_plane_index):
#                 indexall_n = [indexx_n, indexy_n, indexz_n]
#                 indexall_n[i] = indexall[i][planes]

#                 inx, iny, inz = np.meshgrid(indexall_n[0], indexall_n[1], indexall_n[2], indexing='ij')
#                 ind = (inx, iny, inz)

#                 cord1_here = (X[ind])*expand_ratio[0] - n[ind][..., 0]*n_length/2 + origin[0]
#                 cord2_here = (Y[ind])*expand_ratio[1] - n[ind][..., 1]*n_length/2 + origin[1]
#                 cord3_here = (Z[ind])*expand_ratio[2] - n[ind][..., 2]*n_length/2 + origin[2]

#                 nx_here = n[ind][..., 0]
#                 ny_here = n[ind][..., 1]
#                 nz_here = n[ind][..., 2]

#                 if n_color_scalars == True:
#                     scalars_here = n_color_func(n[ind])
#                 else:
#                     scalars_here = X[ind]*0

#                 cord1 = np.concatenate( [ cord1, cord1_here.reshape(-1) ] )
#                 cord2 = np.concatenate( [ cord2, cord2_here.reshape(-1) ] )
#                 cord3 = np.concatenate( [ cord3, cord3_here.reshape(-1) ] )

#                 nx = np.concatenate([nx, nx_here.reshape(-1)])
#                 ny = np.concatenate([ny, ny_here.reshape(-1)])
#                 nz = np.concatenate([nz, nz_here.reshape(-1)])

#                 scalars = np.concatenate([scalars, scalars_here.reshape(-1)])

#         # make plot
#         vector = mlab.quiver3d(
#                 cord1, cord2, cord3,
#                 nx, ny, nz,
#                 mode=n_shape,
#                 scalars=scalars,
#                 scale_factor=n_length,
#                 opacity=n_opacity,
#                 line_width=n_width,
#                 color=n_color
#                 )
#         if n_color_scalars:
#             vector.glyph.color_mode = 'color_by_scalar'
        
#         # make colorbar of n
#         if n_color_scalars == True:
#             if n_if_colorbar:
#                 n_colorbar_label_fmt    = n_colorbar_params.get('label_fmt', '%.2f')
#                 n_colorbar_nb_labels    = n_colorbar_params.get('nb_labels', 5)
#                 n_colorbar_orientation  = n_colorbar_params.get('orientation', 'vertical')
#                 n_lut_manager = mlab.colorbar(object=vector,
#                                                 label_fmt=n_colorbar_label_fmt, 
#                                                 nb_labels=n_colorbar_nb_labels, 
#                                                 orientation=n_colorbar_orientation,
#                                                 **n_colorbar_params)
                
#                 if n_colorbar_range == 'max':
#                     n_colorbar_range = (np.min(scalars), np.max(scalars))
#                 elif n_colorbar_range=='default':
#                     if n_color_func==n_color_func_default:
#                         n_colorbar_range = (0, 2.6)
#                     else:
#                         print('\nWarning: When n_color_func is a function which is not default, and there is n_colorbar')
#                         print('n_colorbar_range should be \'max\', or set by hand, but not \'default\'.')
#                         print('Here n_colorbar_range is automatically turned to be \'max\'.')
#                         print('Read the document for more information')
                    
#                 n_lut_manager.data_range=(n_colorbar_range)

#         elif n_if_colorbar==True:
#             print('\nThe color of n is set. So there is no colorbar for n.')
    
#     # add axes if neede
#     if if_axes:
#         mlab.axes()