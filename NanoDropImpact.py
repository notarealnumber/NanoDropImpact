#!/usr/bin/python3

from get_initial_data import *
from get_basic_data import *
from analysis import *


def main():
    """
    Program to calculate
        1. spreading radius
        2. contact angle
    of a drop after the impact on a solid surface. Data will be writen to file
    and a gnuplot input file will be generated as well.
    For each *.vel file (LAMMPS dump file that contains velocities, coordinates
    and masses) a separate data folder should be created. Something like:
        ./current_dir/
            data01/
            data02/
            data03/
            ...
    depending on the number of *vel files.
    """

    generate_folder_structure()

    dx = 12.0
    dy = 12.0
    dz = 12.0
    drho = 8.0

    # Initiate a few variables
    drop_mass = 0.0
    molar_mass = 0.0
    molecule_weight = 0.0

    calc_radius_angle, datafile, \
        nframes, natoms, dt, \
        calc_vel, max_vel, drop_diameter, \
        clusters, nclusters, cluster_file = get_user_input()

    if calc_radius_angle or calc_vel:
        nmol, tot_nr_of_atoms, \
            first_timestep, \
            xlo, xhi, \
            ylo, yhi, \
            zlo, zhi = get_info(datafile, natoms)

        # Set some constant values and initialize a few arrays, dictionaries and lists that will hold data
        # obtained over the whole trajectory.
        delta_x = xhi - xlo
        delta_y = yhi - ylo
        delta_z = zhi - zlo

        nx = int(np.floor((delta_x + 10.0) / dx))
        ny = int(np.floor((delta_y + 10.0) / dy))
        nz = int(np.floor((delta_z + 10.0) / dz))
        nrho = int(np.floor((np.sqrt(delta_x ** 2 + delta_y ** 2) + 10.0) / drho))

        all_data = []

        if calc_radius_angle:
            density_levels = ()
            for lvl in range(0, 1200, 100):
                density_levels += (lvl,)

        if calc_vel:
            velocity_levels = ()
            for lvl in range(0, max_vel, 100):
                velocity_levels += (lvl,)

        velocity_coord_file = open(datafile, "r")

    if clusters:
        clusters_path = np.full((nframes, nclusters, 3), 0.0)
        clusters_velocity = np.full((nframes, nclusters, 3), 0.0)
        cluster_temp_file = cluster_file[0]
        data_returned = get_info(cluster_temp_file, at_per_mol=1)
        natoms_per_cluster, first_timestep = data_returned[1:3]

        cluster_data_file = []
        for clu in range(nclusters):
            cluster_data_file.append(open(cluster_file[clu], "r"))

    for timestep in range(nframes):
        print("We are at frame:", timestep)

        # all_data.append([(first_timestep + timestep * dt) / 1000.0])
        # Get the raw data first. Velocities of the single atoms, coordinates and atomic masses
        if calc_radius_angle or calc_vel:

            velocities, coordinates, masses = \
                get_raw_data(velocity_coord_file, natoms)

            # Calculate a few constants, just once at the first timestep
            if timestep == 0:
                drop_mass = sum(masses)
                molar_mass = sum(masses[0:natoms])
                molecule_weight = molar_mass / constants.N_A

            # Center of mass of the drop and each molecule, which changes with every timestep
            drop_cofm = get_cofm(
                coordinates, masses,
                drop_mass
            )

            molecules_cofm, velocity_per_molecule = get_molecules_cofm(
                coordinates,
                velocities,
                masses, molar_mass,
                drop_cofm,
                natoms, int(len(coordinates) / natoms),
                xlo, xhi, delta_x,
                ylo, yhi, delta_y,
                calc_vel
            )

            rho_z_molecules_distribution, xyz_molecules_distribution, \
                rho_velocity_distribution, z_velocity_distribution, \
                velocity_abs_value_rhoz, velocity_abs_value_xyz, \
                xmin, xmax, ymin, ymax, zmax, rhomax = \
                get_rhoz_xyz_vel_distribution(
                    molecules_cofm,
                    velocity_per_molecule,
                    drho, dz,
                    nrho, nz,
                    dx, dy,
                    nx, ny,
                    calc_vel
                )

            if calc_radius_angle:
                all_data.append(
                    spread_radius(rho_z_molecules_distribution,
                                  xyz_molecules_distribution,
                                  density_levels,
                                  molecule_weight,
                                  timestep, first_timestep, dt,
                                  drop_diameter,
                                  dx, dy, dz, drho
                                  )
            )

            if calc_vel:
                vel_dens_distribution(rho_z_molecules_distribution,
                                      rho_velocity_distribution,
                                      z_velocity_distribution,
                                      velocity_abs_value_rhoz,
                                      velocity_levels,
                                      timestep, first_timestep, dt,
                                      drop_diameter,
                                      dz, drho
                                      )

        if clusters:
            for clu in range(nclusters):
                cluster_vel, cluster_trajectory, cluster_masses = \
                    get_raw_data(
                        cluster_data_file[clu],
                        1,
                        natoms_per_cluster
                    )

                clusters_path[timestep, clu, :] = \
                    get_cofm(
                        cluster_trajectory,
                        cluster_masses,
                        natoms_per_cluster,
                        sum(cluster_masses[:])
                    )

                clusters_velocity[timestep, clu, :] = \
                    get_per_cluster_vel(
                        cluster_vel, natoms_per_cluster
                    )

    if calc_vel or calc_radius_angle:
        velocity_coord_file.close()
    if clusters:
        for clu in range(nclusters):
            cluster_data_file[clu].close()

    if calc_radius_angle:
        all_data_file = open("./AllData_" +
                             str(first_timestep / 1e03) + "ps-" +
                             str((first_timestep + nframes * dt) / 1e03) + "ps.data",
                             "w"
                             )
        print("# Time given in [ps]. R(2) and theta(3) represent the ",
              file=all_data_file)
        print("# the spreading radius and the contact angle obtained from ",
              file=all_data_file)
        print("# fitting a circle to the density profile of the drop.",
              file=all_data_file)
        print("# xc(5), yc(6), and R(7) represent the circle's center",
              file=all_data_file)
        print("# coordinates and the spreading radius as obtained from a",
              file=all_data_file)
        print("# fit to the extension of the drop in x and y.",
              file=all_data_file)
        print("#   t(1)     R(2) theta(3)    "
              "xc(5)    yc(6)     R(7)    ", file=all_data_file)

        for data in all_data:
            for i in range(len(data)):
                if data[1] == -99999.0:
                    # Error rose earlier, skip writing data
                    continue
                else:
                    print("{:8.3f}".format(data[i]), end=" ", file=all_data_file)
            print(end="\n", file=all_data_file)

        all_data_file.close()

    if clusters:
        write_path(nclusters, clusters_path, first_timestep, dt)
        create_velocities(nclusters, clusters_velocity, first_timestep, dt)


main()
