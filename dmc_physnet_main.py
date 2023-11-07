#!/usr/bin/env python3

# imports
import argparse

from os.path import splitext
import math
import sys
from datetime import datetime
from neural_network.NeuralNetwork import *
from neural_network.activation_fn import *
import time

"""
DMC code for the calculation of zero-point energies using PhysNet based PESs on GPUs/CPUs. The calculation
is performed in Cartesian coordinates.
See e.g. American Journal of Physics 64, 633 (1996); https://doi.org/10.1119/1.18168 for DMC implementation
and https://scholarblogs.emory.edu/bowman/diffusion-monte-carlo/ for Fortran 90 implementation.

run as: python dmc_physnet_main.py @run_fam_cart.inp -i fam_cart_coor.xyz
The file run_fam_cart.inp specifies the DMC simulation settings and NN architecture used.
The file fam_cart_coor.xyz specifies the minimum and a starting geometry for the studied molecule. 
"""

np.set_printoptions(threshold=sys.maxsize)
# parse command line arguments
# define command line arguments for DMC
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument("--natm", type=int, help="number of atoms")
parser.add_argument("--nwalker", type=int, help="")
parser.add_argument("--stepsize", type=float, help="The stepsize in imaginary time (atomic unit)")
parser.add_argument("--nstep", type=int, help="Total number of steps")
parser.add_argument("--eqstep", type=int, help="Number of steps for equilibration")
parser.add_argument("--alpha", type=float, help="Feed-back parameter, usually propotional to 1/stepsize")
parser.add_argument("--fbohr", type=int, default=0,
                    help="1 if the geometry given in the input is in bohr, 0 if angstrom")
# define command line arguments for NN
parser.add_argument("--num_features", type=int, help="dimensionality of feature vectors")
parser.add_argument("--num_basis", type=int, help="number of radial basis functions")
parser.add_argument("--num_blocks", type=int, help="number of interaction blocks")
parser.add_argument("--num_residual_atomic", type=int, help="number of residual layers for atomic refinements")
parser.add_argument("--num_residual_interaction", type=int, help="number of residual layers for the message phase")
parser.add_argument("--num_residual_output", type=int, help="number of residual layers for the output blocks")
parser.add_argument("--cutoff", default=6.0, type=float, help="cutoff distance for short range interactions")
parser.add_argument("--use_electrostatic", default=1, type=int, help="use electrostatics in energy prediction (0/1)")
parser.add_argument("--use_dispersion", default=1, type=int, help="use dispersion in energy prediction (0/1)")
parser.add_argument("--grimme_s6", default=None, type=float, help="grimme s6 dispersion coefficient")
parser.add_argument("--grimme_s8", default=None, type=float, help="grimme s8 dispersion coefficient")
parser.add_argument("--grimme_a1", default=None, type=float, help="grimme a1 dispersion coefficient")
parser.add_argument("--grimme_a2", default=None, type=float, help="grimme a2 dispersion coefficient")

required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input", type=str,
                      help="input file specifying the minimum and staring geometry (similar to xyz format)",
                      required=True)
args = parser.parse_args()

# make script use a single gpu (if there are multiple gpus per node)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


filename, extension = splitext(args.input)
print("input: ", args.input)

# set seed accoring to time
np.random.seed(int(time.time()))

# set up files to track progress and results of DMC simulation.
# The files are named according to the input file.
potfile = open(filename + ".pot", "w")  # to store the reference potential energy
logfile = open(filename + ".log", "w")  # to store information on DMC settings and final results
errorfile = open("defective_" + filename + ".xyz",
                 "w")  # to write defective geometries to file (e.g. caused by holes in the PES)
lastfile = open("configs_" + filename + ".xyz",
                "w")  # to save the last X molecular geometries to an .xyz file for visual inspection.

##############################################
# initialize/prepare all values/objects
##############################################
# Define constants
emass = 1822.88848
auang = 0.5291772083
aucm = 219474.6313710

# read data from input file (formatted similar to .xyz Files, note the
# hashtags in the file)
in_geoms = np.genfromtxt(args.input)[:, 1:]
atom_type = np.loadtxt(args.input, usecols=0, dtype=str)

mass = []
nucl_charge = []
# retrieve mass and atomic number from input file
# if the code is used with a different PES, the loop below can be extended.
for i in range(args.natm):
    if atom_type[i] == 'H':
        mass.append(1.008)
        nucl_charge.append(1)
    elif atom_type[i] == 'C':
        mass.append(12.011)
        nucl_charge.append(6)
    elif atom_type[i] == 'O':
        mass.append(15.999)
        nucl_charge.append(8)
    else:
        print("UNKNOWN LABEL/atom type", atom_type[i])
        quit()

mass = np.array(mass)
mass = np.sqrt(np.array(mass * emass))

# min structure
xmin = in_geoms[:args.natm, :]

# TS for ground state FAD / for FAM I used a distorted geometry
x0 = in_geoms[-args.natm:, :]

# define NN model (file path to checkpoint files), total charge of the system
# and maximum batch size for prediction of energies (this number can be adapted
# and increased based on GPU model/RAM)
checkpoints = "tl_models/tl_866_fad_ccsdt"
Qref = 0.0
max_batch = 6000

natm = args.natm
nwalker = args.nwalker
stepsize = args.stepsize
nstep = args.nstep
eqstep = args.eqstep
alpha = args.alpha
fbohr = args.fbohr


def log_begin(filename):
    """subroutine to write header of log file
       logging all job details
    """
    logfile.write("                  DMC for " + filename + "\n\n")
    logfile.write("DMC Simulation started at " + str(datetime.now()) + "\n")
    logfile.write("Number of random walkers: " + str(nwalker) + "\n")

    logfile.write("Number of total steps: " + str(nstep) + "\n")
    logfile.write("Number of steps before averaging: " + str(eqstep) + "\n")
    logfile.write("Stepsize: " + str(stepsize) + "\n")
    logfile.write("Alpha: " + str(alpha) + "\n\n")


def log_end(filename):
    """function to write footer of logfile
    """
    logfile.write("DMC Simulation terminated at " + str(datetime.now()) + "\n")
    logfile.write("DMC calculation terminated successfully\n")


def record_error(refx, mass, symb, errq, v, idx):
    """function to save defective geometries to 
    """
    auang = 0.5291772083
    aucm = 219474.6313710

    # errx = errq[0]*auang
    # errx = errx.reshape(-1, 3)

    if len(idx[0]) == 1:
        natm = int(len(refx) / 3)
        errx = errq[0] * auang
        errx = errx.reshape(natm, 3)
        errorfile.write(str(int(natm)) + "\n")
        errorfile.write(str(v[idx[0]] * aucm) + "\n")
        for i in range(int(natm)):
            errorfile.write(
                str(symb[i]) + "  " + str(errx[i, 0]) + "  " + str(errx[i, 1]) + "  " + str(errx[i, 2]) + "\n")

    else:
        natm = int(len(refx) / 3)
        errx = errq[0] * auang
        errx = errx.reshape(len(idx[0]), natm, 3)

        for j in range(len(errx)):
            errorfile.write(str(int(natm)) + "\n")
            errorfile.write(str(v[idx[0][j]] * aucm) + "\n")
            for i in range(int(natm)):
                errorfile.write(str(symb[i]) + "  " + str(errx[j, i, 0]) + "  " + str(errx[j, i, 1]) + "  " + str(
                    errx[j, i, 2]) + "\n")


def ini_dmc():
    """initialization of a few needed quantities.
    """

    # define stepssize
    deltax = np.sqrt(stepsize) / mass

    # psips_f keeps track of how many walkers are alive (psips_f[0]) and which ones (psips_f[1:], 1 for alive and 0 for dead)
    psips_f[:] = 1
    psips_f[0] = nwalker
    psips_f[nwalker + 1:] = 0

    # psips keeps track of atomic positions of all walkers
    # is initialized to some molecular geometry defined in the input xyz file
    psips[:, :, 0] = x0[:]

    # reference energy (which is updated throughout the DMC simulation) is initialized to energy of v0, referenced to energy
    # of minimum geometry
    v_ref = v0
    v_ave = 0
    v_ref = v_ref - vmin

    potfile.write("0  " + str(psips_f[0]) + "  " + str(v_ref) + "  " + str(v_ref * aucm) + "\n")
    return deltax, psips, psips_f, v_ave, v_ref


def walk(psips, dx):
    """walk routine performs the diffusion process of the replicas by adding to the
       coordinates of the alive replicas sqrt(deltatau)rho, rho is random number
       from Gaussian distr
    """
    # print(psips.shape)
    dim = len(psips[0, :, 0])
    for i in range(dim):
        x = np.random.normal(size=(len(psips[:, 0, 0])))
        psips[:, i, 1] = psips[:, i, 0] + x * dx[math.ceil((i + 1) / 3.0) - 1]
        # print(psips[:,i-1,1])

    return psips


def gbranch(refx, mass, symb, vmin, psips, psips_f, v_ref, v_tot, nalive):
    """The birth-death criteria for the ground state energy. Note that psips is of shape
       (3*nwalker, 3*natm) as only the progressed coordinates (i.e. psips[:,i,1]) are
       given to gbranch
    """

    birth_flag = 0
    error_checker = 0
    # print(psips.shape) #-> (3*nwalker, 3*natm)
    v_psip = get_batch_energy(psips[:nalive, :], nalive)  # predict energy of all alive walkers.

    # reference energy with respect to minimum energy.
    v_psip = v_psip - vmin

    # check for holes, i.e. check for energies that are lower than the one for the (global) min
    if np.any(v_psip < -1e-5):
        error_checker = 1
        idx_err = np.where(v_psip < -1e-5)
        record_error(refx, mass, symb, psips[idx_err, :], v_psip, idx_err)
        print("defective geometry is written to file")
        # kill defective walker idx_err + one as index 0 is counter of alive walkers
        psips_f[idx_err[0] + 1] = 0  # idx_err[0] as it is some stupid array...

    prob = np.exp((v_ref - v_psip) * stepsize)
    sigma = np.random.uniform(size=nalive)

    if np.any((1.0 - prob) > sigma):
        """test whether one of the walkers has to die given the probabilites
           and then set corresponding energies v_psip to zero as they
           are summed up later.
           geometries with high energies are more likely to die.
        """
        idx_die = np.array(np.where((1.0 - prob) > sigma)) + 1
        psips_f[idx_die] = 0
        v_psip[idx_die - 1] = 0.0

    v_tot = np.sum(v_psip)  # sum energies of walkers that are alive (i.e. fullfill conditions)

    if np.any(prob > 1):
        """give birth to new walkers given the probabilities and update psips, psips_f
           and v_tot accordingly.
        """
        idx_prob = np.array(np.where(prob > 1)).reshape(-1)

        for i in idx_prob:
            if error_checker == 0:

                probtmp = prob[i] - 1.0
                n_birth = int(probtmp)
                sigma = np.random.uniform()

                if (probtmp - n_birth) > sigma:
                    n_birth += 1
                if n_birth > 2:
                    birth_flag += 1

                while n_birth > 0:
                    nalive += 1
                    n_birth -= 1
                    psips[nalive - 1, :] = psips[i, :]
                    psips_f[nalive] = 1
                    v_tot = v_tot + v_psip[i]

            else:
                if np.any(i == idx_err[0]):  # to make sure none of the defective geom are duplicated
                    pass
                else:

                    probtmp = prob[i] - 1.0
                    n_birth = int(probtmp)
                    sigma = np.random.uniform()

                    if (probtmp - n_birth) > sigma:
                        n_birth += 1
                    if n_birth > 2:
                        birth_flag += 1

                    while n_birth > 0:
                        nalive += 1
                        n_birth -= 1
                        psips[nalive - 1, :] = psips[i, :]
                        psips_f[nalive] = 1
                        v_tot = v_tot + v_psip[i]

    error_checker = 0
    return psips, psips_f, v_tot, nalive


def branch(refx, mass, symb, vmin, psips, psips_f, v_ref):
    """The birth-death (branching) process, which follows the diffusion step
    """

    nalive = psips_f[0]
    v_tot = 0.0

    psips[:, :, 1], psips_f, v_tot, nalive = gbranch(refx, mass, symb, vmin, psips[:, :, 1], psips_f, v_ref, v_tot,
                                                     nalive)

    # after doing the statistics in gbranch remove all dead replicas.
    count_alive = 0
    psips[:, :, 0] = 0.0  # just to be sure we dont use "old" walkers

    for i in range(nalive):
        """update psips and psips_f using the number of alive walkers (nalive). 
        """
        if psips_f[i + 1] == 1:
            count_alive += 1
            psips[count_alive - 1, :, 0] = psips[i, :, 1]
            psips_f[count_alive] = 1
    psips_f[0] = count_alive
    psips[:, :, 1] = 0.0  # just to be sure we dont use "old" walkers
    psips_f[count_alive + 1:] = 0  # set everything beyond index count_alive to zero

    # update v_ref
    v_ref = v_tot / psips_f[0] + alpha * (1.0 - 3.0 * psips_f[0] / (len(psips_f) - 1))

    return psips, psips_f, v_ref


def get_batch_energy(coor, batch_size):
    """function to predict energies given the coordinates of the molecule. Depending on the max_batch and nwalkers,
       the energy prediction are done all at once or in multiple iterations. 
    """
    if batch_size <= max_batch:  # predict everything at once
        e = sess.run(energy, feed_dict={R: coor.reshape((-1, 3)) * auang, Z: batch_size * symb,
                                        idx_i: idx_ilarge[:(natm * (natm - 1)) * batch_size],
                                        idx_j: idx_jlarge[:(natm * (natm - 1)) * batch_size],
                                        batch_seg: batch_seglarge[:natm * batch_size]})

    else:
        e = np.array([])
        counter = 0
        for i in range(int(batch_size / max_batch) - 1):
            counter += 1
            # print(i*max_batch, (i+1)*max_batch)
            etmp = sess.run(energy, feed_dict={R: coor[i * max_batch:(i + 1) * max_batch, :].reshape((-1, 3)) * auang,
                                               Z: max_batch * symb, idx_i: idx_ilarge[:(natm * (natm - 1)) * max_batch],
                                               idx_j: idx_jlarge[:(natm * (natm - 1)) * max_batch],
                                               batch_seg: batch_seglarge[:natm * max_batch]})
            e = np.append(e, etmp)

        # calculate missing geom according to batch_size - counter * max_batch
        remaining = batch_size - counter * max_batch
        # print(remaining)
        if remaining < 0:  # just to be sure...
            print("someting went wrong with the loop in get_batch_energy")
            quit()

        etmp = sess.run(energy, feed_dict={R: coor[-remaining:, :].reshape((-1, 3)) * auang, Z: remaining * symb,
                                           idx_i: idx_ilarge[:(natm * (natm - 1)) * remaining],
                                           idx_j: idx_jlarge[:(natm * (natm - 1)) * remaining],
                                           batch_seg: batch_seglarge[:natm * remaining]})
        e = np.append(e, etmp)

    # print("time:  ", time.time() - start_time)
    return e * 0.0367493


############################################################
# setup nn for batch prediction / without ase
dtype = tf.float32

if tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None):
    print("\n===========\nrunning on GPU:", tf.test.gpu_device_name())
else:
    print("\n===========\nrunning on CPU")
print("nwalkers:", nwalker, "\n===========\n")

# a single array of nuclear charges
mockdata = {}
mockdata['Z'] = nucl_charge

# create "look up table" for the needed inputs. This is needed because the
# batch size is not always constant (walkers die).
N = len(mockdata['Z'])
i_ = []
j_ = []
for i in range(N):
    for j in range(N):
        if i != j:
            i_.append(i)
            j_.append(j)
global idx_ilarge
idx_ilarge = []
global idx_jlarge
idx_jlarge = []
global batch_seglarge
batch_seglarge = []
for batch in range(12000):  # largest batchsize
    idx_ilarge += [i + N * batch for i in i_]
    idx_jlarge += [j + N * batch for j in j_]
    batch_seglarge += [batch] * N

# tensorize all inputs
Z = tf.placeholder(tf.int32, shape=[None, ], name="Z")
R = tf.placeholder(dtype, shape=[None, 3], name="R")
idx_i = tf.placeholder(tf.int32, shape=[None, ], name="idx_i")
idx_j = tf.placeholder(tf.int32, shape=[None, ], name="idx_j")
batch_seg = tf.placeholder(tf.int32, shape=[None, ], name="batch_seg")

# load standard physnet
nn = NeuralNetwork(F=args.num_features,
                   K=args.num_basis,
                   sr_cut=args.cutoff,
                   num_blocks=args.num_blocks,
                   num_residual_atomic=args.num_residual_atomic,
                   num_residual_interaction=args.num_residual_interaction,
                   num_residual_output=args.num_residual_output,
                   use_electrostatic=(args.use_electrostatic == 1),
                   use_dispersion=(args.use_dispersion == 1),
                   s6=args.grimme_s6,
                   s8=args.grimme_s8,
                   a1=args.grimme_a1,
                   a2=args.grimme_a2,
                   activation_fn=shifted_softplus,
                   seed=None,
                   scope="neural_network",
                   dtype=tf.float32)

# calculate all necessary quantities (unscaled partial charges, energies, forces)
Ea, Qa, Dij, _ = nn.atomic_properties(Z, R, idx_i, idx_j)
energy = nn.energy_from_atomic_properties(Ea, Qa, Dij, Z, idx_i, idx_j, Qref, batch_seg)

xmin = xmin.reshape(-1)  # xmin is the geometry of the minimum
x0 = x0.reshape(-1)  # x0 is the ts geometry serving as reference

#####################################

# initialize space for walkers
dim = natm * 3
psips_f = np.zeros([3 * nwalker + 1], dtype=int)
deltax = np.zeros([natm], dtype=float)
psips = np.zeros([3 * nwalker, dim, 2], dtype=float)
symb = mockdata['Z']

#####################################

if fbohr == 0:
    x0 = x0 / auang
    xmin = xmin / auang

# code to obtain energies of xmin and x0
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    nn.restore(sess, checkpoints)

    vmin = get_batch_energy(xmin, 1)
    v0 = get_batch_energy(x0, 1)

log_begin(filename)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    nn.restore(sess, checkpoints)

    ##########################################
    # START DMC Simulation
    ##########################################
    v_tot = 0

    import time

    # initialize all variables for dmc trajectory
    deltax, psips, psips_f, v_ave, v_ref = ini_dmc()
    # print(psips_f)

    for i in range(nstep):
        start_time = time.time()
        psips[:psips_f[0], :, :] = walk(psips[:psips_f[0], :, :], deltax)

        psips, psips_f, v_ref = branch(x0, mass, symb, vmin, psips, psips_f, v_ref)
        potfile.write(str(i + 1) + "   " + str(psips_f[0]) + "   " + str(v_ref) + "   " + str(v_ref * aucm) + "\n")

        if i > eqstep:
            v_ave += v_ref

        if i > nstep - 10:  # record the last 10 steps of the DMC simulation for visual inspection.
            for j in range(psips_f[0]):
                lastfile.write(str(natm) + "\n\n")
                for l in range(int(natm)):
                    l = l + 1
                    lastfile.write(str(symb[l - 1]) + "  " + str(psips[j, 3 * l - 3, 0] * auang) + "  " + str(
                        psips[j, 3 * l - 2, 0] * auang) + "  " + str(psips[j, 3 * l - 1, 0] * auang) + "\n")
        if i % 10 == 0:
            print("step:  ", i, "time/step:  ", time.time() - start_time, "nalive:   ", psips_f[0])
    v_ave = v_ave / (nstep - eqstep)

    logfile.write("AVERAGE ENERGY OF TRAJ   " + "   " + str(v_ave) + " hartree   " + str(v_ave * aucm) + " cm**-1\n")

# terminate code and close log/pot files
log_end(filename)
potfile.close()
logfile.close()
errorfile.close()
lastfile.close()
