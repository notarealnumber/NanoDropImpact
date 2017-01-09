import numpy as np


def get_dummy_lines(inputfile):
    """
    Reads and discards the header of each frame, as it is not needed.
    Args:
        inputfile (str): the file to be read.
    Returns:
        nothing
    """

    inputfile.readline()
    inputfile.readline()
    inputfile.readline()
    inputfile.readline()
    inputfile.readline()
    inputfile.readline()
    inputfile.readline()
    inputfile.readline()
    inputfile.readline()


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def get_user_input():
    """ Function that asks the user for filename, how many frames
    the trajectory contains, and the number of atoms a molecule has.

    :return:
    filename
    nsteps
    at_per_molec
    """

    cluster_files = []

    inputfile = "input.txt"
    inputs = open(inputfile, "r")

    # filename, nsteps, dt, drop_diameter are always needed.
    filename = inputs.readline().strip("\n")
    nsteps = int(inputs.readline().strip("\n"))
    dt = float(inputs.readline().strip("\n"))
    drop_diameter = float(inputs.readline().strip("\n"))

    calc_radius_angle = inputs.readline().strip("\n")
    if calc_radius_angle == "True":
        calc_radius_angle = True
        at_per_molec = int(inputs.readline().strip("\n"))
    else:
        calc_radius_angle = False
        at_per_molec = int(inputs.readline().strip("\n"))

    calc_vel = inputs.readline().strip("\n")
    if calc_vel == "True":
        calc_vel = True
        max_vel = int(inputs.readline().strip("\n"))
    else:
        calc_vel = False
        max_vel = int(inputs.readline().strip("\n"))

    clusters = inputs.readline().strip("\n")
    if clusters == "True":
        clusters = True
        nclusters = int(inputs.readline().strip("\n"))
        for i in range(nclusters):
            cluster_files.append(inputs.readline().strip("\n"))
    else:
        clusters = False
        nclusters = int(inputs.readline().strip("\n"))

    return calc_radius_angle, filename, \
        nsteps, \
        at_per_molec, \
        dt, \
        calc_vel, max_vel, \
        drop_diameter, \
        clusters, nclusters, cluster_files


def get_info(inputfile,
             at_per_mol):

    fileread = open(inputfile, "r")

    fileread.readline()
    init_timestep = int(fileread.readline())
    fileread.readline()
    nat = int(fileread.readline())
    fileread.readline()
    xlo, xhi = fileread.readline().strip(" ").split()
    ylo, yhi = fileread.readline().strip(" ").split()
    zlo, zhi = fileread.readline().strip(" ").split()
    fileread.readline()

    fileread.close()

    return int(nat / at_per_mol), nat, init_timestep, float(xlo), float(xhi), \
        float(ylo), float(yhi), float(zlo), float(zhi)


def get_raw_data(datafile, nmol, nat):

    velocities = np.full((nmol * nat, 3), 0.0)
    coordinates = np.full((nmol * nat, 3), 0.0)
    masses = np.full((nmol * nat), 0.0)
    nat_tot = int(nmol * nat)

    get_dummy_lines(datafile)

    for i in range(0, nat_tot):
        velocities[i, 0], velocities[i, 1], velocities[i, 2], \
            coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], \
            masses[i] = datafile.readline().strip(" ").split()

    return velocities, coordinates, masses


def generate_folder_structure():

    import os.path
    import shutil

    if not os.path.exists("./drop_profile"):
        os.mkdir("./drop_profile")
    else:
        shutil.rmtree("./drop_profile")
        os.mkdir("./drop_profile")

    if not os.path.exists("./velocity_distribution"):
        os.mkdir("./velocity_distribution")
    else:
        shutil.rmtree("./velocity_distribution")
        os.mkdir("./velocity_distribution")

    if not os.path.exists("./drop_from_top"):
        os.mkdir("./drop_from_top")
    else:
        shutil.rmtree("./drop_from_top")
        os.mkdir("./drop_from_top")

    if not os.path.exists("./data_collection"):
        os.mkdir("./data_collection")
    else:
        shutil.rmtree("./data_collection")
        os.mkdir("./data_collection")
