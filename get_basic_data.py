import numpy as np


def get_cofm(coordinates,
             masses, drop_mass):

    min_z_coord = 1000.0
    nat_tot = len(coordinates)
    print(nat_tot)

    """
    Get the drop's center of mass.
    """
    cofm_drop = np.full(3, 0.0)
    for i in range(0, nat_tot):
        if coordinates[i, 2] < min_z_coord:
            min_z_coord = coordinates[i, 2]

        cofm_drop = [cofm_drop[k] + coordinates[i, k] * masses[i] for k in range(len(cofm_drop))]

    cofm_drop /= drop_mass
    coordinates[:, 2] -= min_z_coord

    return cofm_drop


def get_per_cluster_vel(velocities, natoms):
    """

    :param velocities:
    :param natoms:
    :return:
    """

    velocities = np.array(velocities)
    cluster_velocities = np.full(3, 0.0)
    for i in range(natoms):
        cluster_velocities[0] = sum(velocities[:, 0])
        cluster_velocities[1] = sum(velocities[:, 1])
        cluster_velocities[2] = sum(velocities[:, 2])

    cluster_velocities /= (float(natoms) * 1e-05)

    return cluster_velocities


def get_molecules_cofm(
        coordinates,
        velocities,
        masses, molecular_mass,
        drop_cofm,
        n_atoms_per_mol, nmol,
        xlo, xhi, delta_x,
        ylo, yhi, delta_y,
        get_vel_per_molecule):

    """
    Calculates the center of mass per molecule. Periodic boundary conditions
    are applied here already, but only in x- and y-direction since in z-direction
    the slab limits the movement of the molecules.

    If desired the velocity per molecule will be evaluated as well. If not an empty list
    will be returned.

    :param coordinates:
    :param velocities:
    :param masses:
    :param molecular_mass:
    :param drop_cofm:
    :param n_atoms_per_mol:
    :param nmol:
    :param xlo:
    :param xhi:
    :param delta_x:
    :param ylo:
    :param yhi:
    :param delta_y:
    :param get_vel_per_molecule:
    :return:
    """

    cofm = []
    velocity_per_molecule = []

    if get_vel_per_molecule:
        for imol in range(nmol):

            x_cofm = 0.0
            y_cofm = 0.0
            z_cofm = 0.0

            vx_cofm = 0.0
            vy_cofm = 0.0
            vz_cofm = 0.0

            for iat in range(n_atoms_per_mol):
                x_cofm += coordinates[imol * n_atoms_per_mol + iat, 0] * \
                    masses[imol * n_atoms_per_mol + iat]
                y_cofm += coordinates[imol * n_atoms_per_mol + iat, 1] * \
                    masses[imol * n_atoms_per_mol + iat]
                z_cofm += coordinates[imol * n_atoms_per_mol + iat, 2] * \
                    masses[imol * n_atoms_per_mol + iat]

                vx_cofm += velocities[imol * n_atoms_per_mol + iat, 0]
                vy_cofm += velocities[imol * n_atoms_per_mol + iat, 1]
                vz_cofm += velocities[imol * n_atoms_per_mol + iat, 2]

            x_cofm /= molecular_mass
            y_cofm /= molecular_mass
            z_cofm /= molecular_mass

            vx_cofm /= (float(n_atoms_per_mol) * 1e-05)
            vy_cofm /= (float(n_atoms_per_mol) * 1e-05)
            vz_cofm /= (float(n_atoms_per_mol) * 1e-05)

            if x_cofm > xhi or x_cofm < xlo:
                x_cofm -= np.floor(x_cofm / delta_x) * delta_x

            if y_cofm > yhi or y_cofm < ylo:
                y_cofm -= np.floor(y_cofm / delta_y) * delta_y

            x_cofm -= drop_cofm[0]
            y_cofm -= drop_cofm[1]

            """Centers of mass for each molecule:"""
            cofm.append([x_cofm, y_cofm, z_cofm])
            velocity_per_molecule.append([vx_cofm, vy_cofm, vz_cofm])

    else:
        for imol in range(nmol):

            """
            Calculates the center of mass per molecule. Periodic boundary conditions
            are applied here already, but only in x- and y-direction since in z-direction
            the slab limits the movement of the molecules.
            """

            x_cofm = 0.0
            y_cofm = 0.0
            z_cofm = 0.0

            for iat in range(n_atoms_per_mol):
                x_cofm += coordinates[imol * n_atoms_per_mol + iat, 0] * \
                    masses[imol * n_atoms_per_mol + iat]
                y_cofm += coordinates[imol * n_atoms_per_mol + iat, 1] * \
                    masses[imol * n_atoms_per_mol + iat]
                z_cofm += coordinates[imol * n_atoms_per_mol + iat, 2] * \
                    masses[imol * n_atoms_per_mol + iat]

            x_cofm /= molecular_mass
            y_cofm /= molecular_mass
            z_cofm /= molecular_mass

            if x_cofm > xhi or x_cofm < xlo:
                x_cofm -= np.floor(x_cofm / delta_x) * delta_x

            if y_cofm > yhi or y_cofm < ylo:
                y_cofm -= np.floor(y_cofm / delta_y) * delta_y

            x_cofm -= drop_cofm[0]
            y_cofm -= drop_cofm[1]

            """Centers of mass for each molecule:"""
            cofm.append([x_cofm, y_cofm, z_cofm])

    return cofm, velocity_per_molecule


def get_rhoz_xyz_vel_distribution(cofm_per_molecule,
                                  vel_per_molecule,
                                  drho, dz,
                                  nrho, nz,
                                  dx, dy,
                                  nx, ny,
                                  get_vel_dist):

    from get_initial_data import cart2pol

    """

    :param cofm:
    :param drho:
    :param dz:
    :param nrho:
    :param nz:
    :param dx:
    :param dy:
    :param nx:
    :param ny:
    :return:
    """

    get_xyz_number_distrib = np.full((nx + 50, ny + 50, nz), 0.0)
    get_rho_z_number_distrib = np.full((nrho, nz), 0.0)

    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    zmax = 0
    rhomax = 0

    if get_vel_dist:
        get_rho_vel_distrib = np.full((nrho, nz), 0.0)
        get_z_vel_distrib = np.full((nrho, nz), 0.0)
        get_vel_abs_rho_z = np.full((nrho, nz), 0.0)
        get_vel_abs_xyz = np.full((nx + 50, ny + 50, nz), 0.0)

        for mol in range(len(cofm_per_molecule)):

            rho, theta = cart2pol(
                cofm_per_molecule[mol][0],
                cofm_per_molecule[mol][1]
            )

            irho = int(np.floor(rho / drho))
            ix = int(np.floor(cofm_per_molecule[mol][0] / dx))
            iy = int(np.floor(cofm_per_molecule[mol][1] / dy))
            iz = int(np.floor(cofm_per_molecule[mol][2] / dz))

            # Calculate the absolute value of the velocity of each
            # molecule. Used for contour plots.
            velocity_abs_value = np.sqrt(
                vel_per_molecule[mol][0]**2 +
                vel_per_molecule[mol][1]**2 +
                vel_per_molecule[mol][2]**2
            )

            # Get the polar coordinates of the velocity vector
            velocity_in_plane, phi = cart2pol(
                vel_per_molecule[mol][0] / velocity_abs_value,
                vel_per_molecule[mol][1] / velocity_abs_value
            )

            # Calculate the part of the velocity in rho only.
            velocity_in_rho = velocity_in_plane * \
                              np.cos(np.deg2rad(abs(theta - phi)))

            if ix < xmin:
                xmin = ix
            elif ix > xmax:
                xmax = ix
            if iy < ymin:
                ymin = iy
            elif iy > ymax:
                ymax = iy
            if iz > zmax:
                zmax = iz
            if irho > rhomax:
                rhomax = irho

            get_xyz_number_distrib[ix, iy, iz] += 1.0
            get_rho_z_number_distrib[irho, iz] += 1.0

            get_rho_vel_distrib[irho, iz] += velocity_in_rho
            get_z_vel_distrib[irho, iz] += vel_per_molecule[mol][2] / velocity_abs_value

            get_vel_abs_rho_z[irho, iz] += velocity_abs_value
            get_vel_abs_xyz[ix, iy, iz] += velocity_abs_value

    else:
        get_rho_vel_distrib = np.full((1, 1), 0.0)
        get_z_vel_distrib = np.full((1, 1), 0.0)
        get_vel_abs_rho_z = np.full((1, 1), 0.0)
        get_vel_abs_xyz = np.full((1, 1, 1), 0.0)

        for mol in range(len(cofm_per_molecule)):
            rho, theta = cart2pol(cofm_per_molecule[mol][0], cofm_per_molecule[mol][1])
            irho = int(np.floor(rho / drho))
            ix = int(np.floor(cofm_per_molecule[mol][0] / dx))
            iy = int(np.floor(cofm_per_molecule[mol][1] / dy))
            iz = int(np.floor(cofm_per_molecule[mol][2] / dz))

            if ix < xmin:
                xmin = ix
            elif ix > xmax:
                xmax = ix
            if iy < ymin:
                ymin = iy
            elif iy > ymax:
                ymax = iy
            if iz > zmax:
                zmax = iz
            if irho > rhomax:
                rhomax = irho

            get_xyz_number_distrib[ix, iy, iz] += 1.0
            get_rho_z_number_distrib[irho, iz] += 1.0

    return get_rho_z_number_distrib, \
        get_xyz_number_distrib, \
        get_rho_vel_distrib, \
        get_z_vel_distrib, \
        get_vel_abs_rho_z, \
        get_vel_abs_xyz, \
        xmin, xmax, ymin, ymax, zmax, rhomax
