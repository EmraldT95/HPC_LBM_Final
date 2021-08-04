import numpy as np
from numpy.lib.format import dtype_to_descr, magic
import time
from mpi4py import MPI

# Getting the MPI WORLD which has all the info. about all the processes running in parallel
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

NX = 300                                                # X dimension
NY = 300                                                # Y dimension

def divide_grid(x_, y_):
    prod = x_ * y_
    if prod == size:
        # Check if the grid can be reasonably divided by the shape
        splitX = (NX - ((NX//x_) * x_)) % y_
        splitY = (NY - ((NY//y_) * y_)) % x_
        if splitX == 0 and splitY == 0:
            return x_, y_
        else:
            x_, y_ = divide_grid(x_+1,y_-1)
    elif prod > size:
        x_, y_ = divide_grid(x_,y_-1)
    else:
        x_, y_ = divide_grid(x_+1,y_)
    return x_, y_
sects_X, sects_Y = divide_grid(int(np.sqrt(size)),int(np.sqrt(size))) # No. of sections across X and Y direction
if rank == 0:
    print('Optimal sections cut: X = {}, Y = {}'.format(sects_X, sects_Y))

'''
Cartesian communicator - 
    @param dims: how we want to divide the lattice grid
    @param periods: if there must be periodic flow from one end to the other
'''
cart_comm = comm.Create_cart(dims=(sects_X, sects_Y), periods=[False,False],reorder=False)
rcoords = cart_comm.Get_coords(rank)                    # Coordinates in the cartesian
boundary=[False,False,False,False]                      # Top, Right, Bottom, Left

# Dividing the grid into sections for each processor to deal with
subDom_X = int(NX//sects_X)
subDom_Y = int(NY//sects_Y)

'''
If the no. of processors assigned is not a squared number, then we need to 
make the edge sections larger than the other sections. Here we do this of the Right-most
and Top-most sections
''' 
if rcoords[0] == sects_X - 1:
    subDom_X += int(NX%sects_X)
    boundary[1] = True                                  # Right boundary
if rcoords[0] == 0:
    boundary[3] = True                                  # Left boundary
if rcoords[1] == sects_Y - 1:
    subDom_Y += int(NY%sects_Y)
    boundary[0] = True                                  # Top boundary
if rcoords[1] == 0:
    boundary[2] = True                                  # Bottom boundary

# Define the subdomain bounds
nx1 = int(rcoords[0]*subDom_X)
nx2 = int((rcoords[0]+1)*subDom_X)
ny1 = int(rcoords[1]*subDom_Y)
ny2 = int((rcoords[1]+1)*subDom_Y)
print('Rank {} {}: Sub-domain {}, {}'.format(rank,rcoords,subDom_X,subDom_Y))

# Adding buffer nodes for each section
subDom_X += 2
subDom_Y += 2

'''
Get the neighboring sub-domains to decide where to receive from and where send to.
Not using shift logic as get diagonal neighbors is not possible using shift
''' 
TOP_n = -2
BOTTOM_n = -2
LEFT_n = -2
RIGHT_n = -2
TOP_RIGHT_n = -2
TOP_LEFT_n = -2
BOTTOM_RIGHT_n = -2
BOTTOM_LEFT_n = -2

print('Rank {} {}: Boundaries {}'.format(rank,rcoords,boundary))
if not boundary[0]:
    TOP_n = cart_comm.Get_cart_rank([rcoords[0], rcoords[1]+1])                 # TOP
    if not boundary[1]:
        TOP_RIGHT_n = cart_comm.Get_cart_rank([rcoords[0]+1, rcoords[1]+1])     # TOP-RIGHT
    if not boundary[3]:
        TOP_LEFT_n = cart_comm.Get_cart_rank([rcoords[0]-1, rcoords[1]+1])      # TOP-LEFT
if not boundary[2]:
    BOTTOM_n = cart_comm.Get_cart_rank([rcoords[0], rcoords[1]-1])              # BOTTOM
    if not boundary[1]:
        BOTTOM_RIGHT_n = cart_comm.Get_cart_rank([rcoords[0]+1, rcoords[1]-1])  # BOTTOM-RIGHT
    if not boundary[3]:
        BOTTOM_LEFT_n = cart_comm.Get_cart_rank([rcoords[0]-1, rcoords[1]-1])   # BOTTOM-LEFT
if not boundary[1]:
    RIGHT_n = cart_comm.Get_cart_rank([rcoords[0]+1, rcoords[1]])               # RIGHT
if not boundary[3]:
    LEFT_n = cart_comm.Get_cart_rank([rcoords[0]-1, rcoords[1]])                # LEFT

print('Rank {} {}: T {}, L {}, B {}, R {}, TR {}, TL {}, BR {}, BL {}'.format(
        rank,rcoords,TOP_n,LEFT_n,BOTTOM_n,RIGHT_n,TOP_RIGHT_n,TOP_LEFT_n,BOTTOM_RIGHT_n,BOTTOM_LEFT_n))
'''******************************
*        Common Variables       *
******************************'''

omega = 1.7                                                 # The relaxation time 1/T
w_i = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]     # Weights
c_ij = np.array([[0,0],[1,0],[0,1],                         # Channel-wise velocitz vectors
                 [-1,0],[0,-1],[1,1],
                 [-1,1],[-1,-1],[1,-1]])
nu = 1/3 * (1/omega - 1/2)                                  # Kinectic viscosity

'''******************************
*        Common Functions       *
******************************'''

# Function to find the equilibrium distribution function
'''
Function to find the equilibrium distribution function
    @param(2D array) rho_ij_: the density of all the lattice points
    @param(3D array) u_ijk_: the velocities of all lattice points - (u_x, u_y)
'''
def calc_feq(rho_ij_, u_ijk_):
    eq_f_star = np.ones([9,subDom_Y,subDom_X])
    u2_ij = np.einsum('ijk -> jk', u_ijk_ * u_ijk_)
    # Loop through all the channels
    for i in range(0,9):
        cu_ij = np.einsum('n, njk -> jk', c_ij[i], u_ijk_)
        cu2_ij = np.einsum('ij -> ij', cu_ij * cu_ij)
        eq_f_star[i] = w_i[i] * rho_ij_ * (1 + 3 * cu_ij + (9/2) * cu2_ij - (3/2) * u2_ij)
    return eq_f_star

'''
Streaming function - 
    @param(3D array) f_ijk_: the probability density in this sub-domain
'''
def streaming(f_ijk_):
    for i in range(1,9):
        f_ijk_[i] = np.roll(f_ijk_[i,:,:], shift=c_ij[i], axis=(0,1))
    return f_ijk_

'''
Bounce-Back function - 
    @param(3D array) f_ijk_: the probability density in this sub-domain
    @param(1D array) u_wall: velocity of the wall(s) - [TOP, RIGHT, BOTTOM, LEFT]
    @param(1D array) rho_wall: density near the wall(s) - [TOP, RIGHT, BOTTOM, LEFT]
'''
def bounce_back(f_ijk_, u_wall=[], rho_wall=[]):

    # If TOP boundary exists
    if boundary[0]:
        f_ijk_[4, :, -2] = f_ijk_[2, :, -1]
        # If moving TOP boundary
        if u_wall[0][0] > 0:
            f_ijk_[7, :, -2] = np.roll(f_ijk_[5, :, -1], -1) - 6. * w_i[5] * rho_wall[0] * u_wall[0][0]
            f_ijk_[8, :, -2] = np.roll(f_ijk_[6, :, -1], 1) + 6. * w_i[6] * rho_wall[0] * u_wall[0][0]
        else:
            f_ijk_[7, :, -2] = np.roll(f_ijk_[5, :, -1], -1)
            f_ijk_[8, :, -2] = np.roll(f_ijk_[6, :, -1], 1)
    
    # If RIGHT boundary exists
    if boundary[1]:
        f_ijk_[3, -2, :] = f_ijk_[1, -1, :]
        # If moving RIGHT boundary
        if u_wall[1][1] > 0:
            f_ijk_[6, -2, :] = np.roll(f_ijk_[8, -1, :], 1) - 6. * w_i[8] * rho_wall[1] * u_wall[1][1]
            f_ijk_[7, -2, :] = np.roll(f_ijk_[5, -1, :], -1) + 6. * w_i[5] * rho_wall[1] * u_wall[1][1]
        else:
            f_ijk_[6, -2, :] = np.roll(f_ijk_[8, -1, :], 1)
            f_ijk_[7, -2, :] = np.roll(f_ijk_[5, -1, :], -1)

    # If BOTTOM boundary exists
    if boundary[2]:
        f_ijk_[2, :, 1] = f_ijk_[4, :, 0]
        # If moving BOTTOM boundary
        if u_wall[2][0] > 0:
            f_ijk_[5, :, 1] = np.roll(f_ijk_[7, :, 0], 1) - 6. * w_i[7] * rho_wall[2] * u_wall[2][0]
            f_ijk_[6, :, 1] = np.roll(f_ijk_[8, :, 0], -1) + 6. * w_i[8] * rho_wall[2] * u_wall[2][0]
        else:
            f_ijk_[5, :, 1] = np.roll(f_ijk_[7, :, 0], 1)
            f_ijk_[6, :, 1] = np.roll(f_ijk_[8, :, 0], -1)
            
    # If LEFT boundary exists
    if boundary[3]:
        f_ijk_[1, 1, :] = f_ijk_[3, 0, :]
        # If moving LEFT boundary
        if u_wall[3][1] > 0:
            f_ijk_[5, 1, :] = np.roll(f_ijk_[7, 0, :], 1) - 6. * w_i[7] * rho_wall[3] * u_wall[3][1]
            f_ijk_[8, 1, :] = np.roll(f_ijk_[6, 0, :], -1) + 6. * w_i[6] * rho_wall[3] * u_wall[3][1]
        else:
            f_ijk_[5, 1, :] = np.roll(f_ijk_[7, 0, :], 1)
            f_ijk_[8, 1, :] = np.roll(f_ijk_[6, 0, :], -1)

    return f_ijk_

'''
Collision function - 
    @param(3D array) f_ijk_: the probability density in this sub-domain
'''
def collision(f_ijk_):
    # Calculating the density and velocity
    rho_new = np.einsum('ijk->jk', f_ijk_)
    u_new = np.einsum('ji, ikl -> jkl', c_ij.T, f_ijk_)/rho_new
    # Calculating the equilibrium probability density
    eq_f_ijk = calc_feq(rho_new, u_new)
    f_star = f_ijk_ + omega * (eq_f_ijk - f_ijk_)
    return f_star, rho_new, u_new

'''
Write a global two-dimensional array to a single file in the npy format
using MPI I/O: https://docs.scipy.org/doc/numpy/neps/npy-format.html
Arrays written with this function can be read with numpy.load - 
    @param(String) fn: the File name
    @param(array_like) g_kl: Portion of the array on this MPI processes
'''
def save_mpiio(comm, fn, g_kl):
    magic_str = magic(1, 0)

    local_nx, local_ny = g_kl.shape
    nx = np.empty_like(local_nx)
    ny = np.empty_like(local_ny)

    commx = comm.Sub((True, False))
    commy = comm.Sub((False, True))
    commx.Allreduce(np.asarray(local_nx), nx)
    commy.Allreduce(np.asarray(local_ny), ny)

    arr_dict_str = str({ 'descr': dtype_to_descr(g_kl.dtype),
                         'fortran_order': False,
                         'shape': (np.asscalar(nx), np.asscalar(ny)) })
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += ' '
    arr_dict_str += '\n'
    header_len = len(arr_dict_str) + len(magic_str) + 2

    offsetx = np.zeros_like(local_nx)
    commx.Exscan(np.asarray(ny*local_nx), offsetx)
    offsety = np.zeros_like(local_ny)
    commy.Exscan(np.asarray(local_ny), offsety)

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if rank == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode('latin-1'))
    mpitype = MPI._typedict[g_kl.dtype.char]
    filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
    filetype.Commit()
    file.Set_view(header_len + (offsety+offsetx)*mpitype.Get_size(),
                  filetype=filetype)
    file.Write_all(g_kl.copy())
    filetype.Free()
    file.Close()

'''
Function that sends to and receives from other processes. Here we have to take care of 
of a few things like possible deadlocks and memory allocation. Since the memory for each
sub-domain is stored contiguisly in a column-wise fashion, we need to create copies of the
TOP and BOTTOM rows and store them in buffer variables so that accessing the values would be a lot
more cheaper.
    @param(3D array) f_ijk_: the probability density in this sub-domain
'''
def Send_Recv(f_ijk_):
    '''
    Communication to/from the LEFT and RIGHT neighbours for all relevant channels
    '''
    # If RIGHT neighbour exists
    if RIGHT_n != -2:
        # If LEFT neighbour exists as well
        if LEFT_n != -2:
            # Channel 1,5,8 => Send(RIGHT) and Receive(LEFT)
            f_ijk_[[1,5,8],0,:] = comm.sendrecv(sendobj=f_ijk_[[1,5,8],-2,:], dest=RIGHT_n, source=LEFT_n)
            # Channel 3,6,7 => Send(LEFT) and Receive(RIGHT)
            f_ijk_[[3,6,7],-1,:] = comm.sendrecv(sendobj=f_ijk_[[3,6,7],1,:], dest=LEFT_n, source=RIGHT_n)

        # If only RIGHT neighbour exists
        else:
            # Channel 1,5,8 => Send(RIGHT)
            # Channel 3,6,7 => Receive(RIGHT)
            f_ijk_[[3,6,7],-1,:] = comm.sendrecv(sendobj=f_ijk_[[1,5,8],-2,:], dest=RIGHT_n, source=RIGHT_n)

    # If only LEFT neighbour exists
    elif LEFT_n != -2:
        # Channel 3 => Send(LEFT)
        # Channel 1 => Receive(LEFT)
        f_ijk_[[1,5,8],0,:] = comm.sendrecv(sendobj=f_ijk_[[3,6,7],1,:], dest=LEFT_n, source=LEFT_n)

    '''
    Communication to/from the TOP and BOTTOM neighbours for all relevant channels
    Here we use buffers to store the data in a contiguous manner
    '''
    send_buffer_256 = f_ijk_[[2,5,6],:,-2].copy()  # Channel 2,5,6 => copy TOP row
    send_buffer_478 = f_ijk_[[4,7,8],:,1].copy()   # Channel 4,7,8 => copy BOTTOM row
    # If TOP neighbour exists
    if TOP_n != -2:
        # If BOTTOM neighbour exists as well
        if BOTTOM_n != -2:
            # Channel 2,5,6 => Send(TOP) and Receive(BOTTOM)
            receive_buffer = np.zeros_like(f_ijk_[[2,5,6],:,0])
            cart_comm.Sendrecv(sendbuf=send_buffer_256, dest=TOP_n, recvbuf=receive_buffer, source=BOTTOM_n)
            f_ijk_[[2,5,6],:,0] = receive_buffer       # Channel 2,5,6 => store data in receive_buffer
            # Channel 4,7,8 => Send(BOTTOM) and Receive(TOP)
            receive_buffer = np.zeros_like(f_ijk_[[4,7,8],:,-1])            
            cart_comm.Sendrecv(sendbuf=send_buffer_478, dest=BOTTOM_n, recvbuf=receive_buffer, source=TOP_n)
            f_ijk_[[4,7,8],:,-1] = receive_buffer      # Channel 4,7,8 => store data in receive_buffer to TOP row

        # If only TOP neighbour exists
        else:
            # Channel 2 => Send(TOP)
            # Channel 4 => Receive(TOP)
            receive_buffer = np.zeros_like(f_ijk_[[4,7,8],:,-1]) 
            cart_comm.Sendrecv(sendbuf=send_buffer_256, dest=TOP_n, recvbuf=receive_buffer, source=TOP_n)
            f_ijk_[[4,7,8],:,-1] = receive_buffer      # Channel 4 => store data in receive_buffer to TOP row

    # If only BOTTOM neighbour exists
    elif BOTTOM_n != -2:
        # Channel 4 => Send(BOTTOM)
        # Channel 2 => Receive(BOTTOM)
        receive_buffer = np.zeros_like(f_ijk_[[2,5,6],:,0])
        cart_comm.Sendrecv(sendbuf=send_buffer_478, dest=BOTTOM_n, recvbuf=receive_buffer, source=BOTTOM_n)
        f_ijk_[[2,5,6],:,0] = receive_buffer           # Channel 2 => store data in receive_buffer to BOTTOM row

    '''
    Communication to/from the TOP_RIGHT, TOP_LEFT, BOTTOM_RIGHT and BOTTOM_LEFT neighbours
    for all relevant channels
    '''
    # If TOP_RIGHT neighbour exists
    if TOP_RIGHT_n != -2:
        # If BOTTOM_LEFT neighbour exists as well
        if BOTTOM_LEFT_n != -2:
            # Channel 5 => Send(TOP_RIGHT) and Receive(BOTTOM_LEFT)
            f_ijk_[5,0,0] = comm.sendrecv(sendobj=f_ijk_[5,-2,-2], dest=TOP_RIGHT_n, source=BOTTOM_LEFT_n)
             # Channel 7 => Send(BOTTOM_LEFT) and Receive(TOP_RIGHT)
            f_ijk_[7,-1,-1] = comm.sendrecv(sendobj=f_ijk_[7,1,1], dest=BOTTOM_LEFT_n, source=TOP_RIGHT_n)
        # If only TOP_RIGHT neighbour exists        
        else:
            # Channel 5 => Send(TOP_RIGHT)
            # Channel 7 => Receive(TOP_RIGHT)
            f_ijk_[7,-1,-1] = comm.sendrecv(sendobj=f_ijk_[5,-2,-2], dest=TOP_RIGHT_n, source=TOP_RIGHT_n)
    # If only BOTTOM_LEFT neighbour exists
    elif BOTTOM_LEFT_n != -2:
        # Channel 7 => Send(BOTTOM_LEFT)
        # Channel 5 => Receive(BOTTOM_LEFT)
        f_ijk_[5,0,0] = comm.sendrecv(sendobj=f_ijk_[7,1,1], dest=BOTTOM_LEFT_n, source=BOTTOM_LEFT_n)
        
    # If TOP_LEFT neighbour exists
    if TOP_LEFT_n != -2:
        # If BOTTOM_RIGHT neighbour exists as well
        if BOTTOM_RIGHT_n != -2:
            # Channel 6 => Send(TOP_LEFT) and Receive(BOTTOM_RIGHT)
            f_ijk_[6,-1,0] = comm.sendrecv(sendobj=f_ijk_[6,1,-2], dest=TOP_LEFT_n, source=BOTTOM_RIGHT_n)
            # Channel 8 => Send(BOTTOM_RIGHT) and Receive(TOP_LEFT)
            f_ijk_[8,0,-1] = comm.sendrecv(sendobj=f_ijk_[8,-2,1], dest=BOTTOM_RIGHT_n, source=TOP_LEFT_n)
        # If only TOP_LEFT neighbour exists
        else:
            # Channel 6 => Send(TOP_LEFT)
            # Channel 8 => Receive(TOP_LEFT)
            f_ijk_[8,0,-1] = comm.sendrecv(sendobj=f_ijk_[6,1,-2], dest=TOP_LEFT_n, source=TOP_LEFT_n)
    # If only BOTTOM_RIGHT neighbour exists
    elif BOTTOM_RIGHT_n != -2:
        # Channel 8 => Send(BOTTOM_RIGHT)
        # Channel 6 => Receive(BOTTOM_RIGHT)
        f_ijk_[6,-1,0] = comm.sendrecv(sendobj=f_ijk_[8,-2,1], dest=BOTTOM_RIGHT_n, source=BOTTOM_RIGHT_n)
    
    return f_ijk_

'''******************************
#    Variable Initializations   *
******************************'''
rho_ij = np.ones([subDom_Y, subDom_X])                                  # Density at each lattice point
u_ijk = np.zeros([2, subDom_Y, subDom_X])                               # Velocity at each lattice point
# Density - [TOP, RIGHT, BOTTOM, LEFT]
rho_wall = [1.0, 1.0, 1.0, 1.0]
# Wall velocities - [TOP, RIGHT, BOTTOM, LEFT]
u_wall = [[0.1, 0.], [0., 0.], [0., 0.], [0., 0.]]

# Initial probability distribution
f_ijk = calc_feq(rho_ij, u_ijk)

'''******************************
#          Sliding Lid          *
******************************'''
boundary_exists = boundary[0] or boundary[1] or boundary[2] or boundary[3]
startTime = time.time()

try:
    for i in np.arange(100000):
        # Communicate with all neighbours
        f_ijk = Send_Recv(f_ijk)
        # Only moving TOP wall. So find density near the wall (no velocity in y-direction, i.e, u_y = 0)
        rho_wall[0] = f_ijk[0,:,-2]+f_ijk[1,:,-2]+2.*f_ijk[2,:,-2]+f_ijk[3,:,-2]+2.*f_ijk[5,:,-2]+2.*f_ijk[6,:,-2]
        # Streaming
        f_ijk = streaming(f_ijk)
        # Bounce back (If atleast one of the boundaries exist)
        if (boundary_exists):
            f_ijk = bounce_back(f_ijk, u_wall, rho_wall)
        # Collision
        f_ijk, rho_ij, u_ijk = collision(f_ijk)

    if rank == 0:
        print('No. of lattice points = {}, Iterations = {}, Approx. Total execution time = {}'.format(NX * NY, 100000, (time.time() - startTime) * size))
    print('Rank {}: Saving data...'.format(rank))
    save_mpiio(cart_comm, 'ux.npy', u_ijk[0])
    save_mpiio(cart_comm, 'uy.npy', u_ijk[1])
    print('Rank {}: Save Complete'.format(rank))

except(Exception, ArithmeticError) as e:
    print(e)
