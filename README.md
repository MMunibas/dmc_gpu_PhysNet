GPU implementation of the diffusion monte carlo algorithm for use with PhysNet interatomic potentials 


                       Diffusion Monte Carlo Code for PhysNet
                         Meuwly Group, University of Basel                  


### General

The Diffusion Monte Carlo (DMC) Code here can be used to calculate the vibrational 
zero-point energy of molecular systems. It is based on the unbiased DMC algorithm
detailed in Ref [1] and uses PhysNet [2] based potential energy surfaces (PESs) to
obtain energies for molecular geometries. It is recommended to run the calculations
on a GPU for higher efficiency.

### Installations & dependencies

The following installation steps were tested on a Ubuntu 18.04 workstation and using
Conda 4.12.0 (see https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

a) Create an environment named (e.g.) physnet_dmc_env, install Python 3.6:

    conda create --name physnet_dmc_env python=3.6

   Activate it:

    conda activate physnet_dmc_env

    (deactivating it by typing: conda deactivate)


b) install tensorflow (and all dependencies) using conda:

        conda install tensorflow-gpu==1.13.1

    if a compatible gpu is available, else try:

        conda install tensorflow==1.13.1

### Example

The use of the code is illustrated based on the formic acid monomer molecule, for
which a PhysNet PES has been published [3] (also see https://github.com/MMunibas/PhysNet-formic-acid-PES).


The repository contains the following files and folders:

i) dmc_physnet_main.py: Contains the DMC code

ii) fam_cart_coor.xyz: Contains the equilibrium FAM geometry and a reference geometry

iii) run_fam_cart.inp: Contains the DMC simulation settings and NN architecture used

iv) tl_models: Contains the NN models that are transfer learnt to the CCSD(T)/aVTZ level of theory

v) neural_network: Contains the PhysNet implementation


The code can be run by invoking:

    python dmc_physnet_main.py @run_fam_cart.inp -i fam_cart_coor.xyz

The following files are produced by the DMC code:

i) fam_cart_coor.log: Summarizes the DMC settings and gives the final result of the simulation (ZPE of FAM on the PES is around 7320 cm**-1, see Ref. [3])

ii) fam_cart_coor.pot: Keeps track of the reference energy and number of alive walkers throughout the simulation.

iii) configs_fam_cart_coor.xyz: Saves the walkers from the last 10 DMC steps for visualization purposes (in .xyz format)

iv) defective_fam_cart_coor.xyz: Saves walkers that are defective, i.e. from regions on the PES that were insufficiently sampled (holes).

### How to cite 

When using the PhysNet PES for FAM and the DMC code, please cite the following papers:

Unke, O. T. and Meuwly, M "PhysNet: A Neural Network for Predicting Energies,
Forces, Dipole Moments, and Partial Charges", J. Chem. Theory Comput. 2019,
15, 6, 3678–3693

Käser, S. and Meuwly, M. "Transfer learned potential energy surfaces: accurate
anharmonic vibrational dynamics and dissociation energies for the formic acid
monomer and dimer", Phys. Chem. Chem. Phys., 2022, 24, 5269-5281.

### References

[1] Ioan Kosztin, Byron Faber, and Klaus Schulten; Am. J. Phys. 64, 633 (1996); https://doi.org/10.1119/1.18168

[2] Oliver T. Unke and Markus Meuwly; J. Chem. Theory Comput. 2019, 15, 6, 3678–3693

[3] Käser, S. and Meuwly, M.; Phys. Chem. Chem. Phys., 2022, 24, 5269-5281.

### Contact

If you have any questions about the PES free to contact Silvan Kaeser
(silvan.kaeser@unibas.ch)

