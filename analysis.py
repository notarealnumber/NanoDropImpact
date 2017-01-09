import scipy.interpolate
from least_square_circle import *
import numpy as np
from scipy import odr


def spread_radius(rho_z_distribution,
                  xyz_distribution,
                  levels,
                  m_molecule,
                  iframe, first_timestep, dt,
                  diameter,
                  dx, dy, dz, drho
                  ):

    """

    :param rho_z_distribution:
    :param xyz_distribution:
    :param levels:
    :param m_molecule:
    :param iframe:
    :param first_timestep:
    :param dt:
    :param diameter:
    :param dx:
    :param dy:
    :param dz:
    :param drho:
    :return:
    """

    z_height = 2

    size_rho = int(np.ceil(diameter / drho)) + 2
    size_z = size_rho
    rho_rho = []
    rho_z = []
    rho_density = []
    all_data = [(first_timestep + iframe * dt) / 1e03]

    for iz in range(size_z):
        for irho in range(size_rho):
            volume = (np.pi * ((irho + 1) * drho)**2 -
                      np.pi * (irho * drho)**2) * dz * 1.0e-30
            density = rho_z_distribution[irho, iz] / volume * m_molecule / 1000.0
            rho_rho.append(irho * drho)
            rho_z.append(iz * dz)
            rho_density.append(density)

    rho_rho = np.array(rho_rho)
    rho_z = np.array(rho_z)
    rho_density = np.array(rho_density)

    try:

        xi, yi = np.linspace(rho_rho.min(), rho_rho.max(), 1000), \
            np.linspace(rho_z.min(), rho_z.max(), 1000)
        xi, yi = np.meshgrid(xi, yi)

        zi = scipy.interpolate.griddata((rho_rho, rho_z), rho_density, (xi, yi), method='cubic')

        for k in range(len(zi[1])):
            for j in range(len(zi[0])):
                if zi[j, k] < 0.0:
                    zi[j, k] = 0.0

        fig = plt.figure()
        ax = fig.gca()

        cntrf = plt.contour(
            xi, yi, zi,
            levels[1:-1],
            colors='k',
            fontproperties='Open Sans'
        )

        cntr = plt.contourf(
            xi,
            yi,
            zi,
            levels[0:-1],
            vmin=levels[0], vmax=levels[-1],
            origin='lower',
            extent=[
                rho_rho.min(), rho_rho.max(),
                rho_z.min(), rho_z.max()]
        )

        vertices_for_fit = []
        for i in range(len(cntrf.collections[2].get_paths())):
            data = cntrf.collections[2].get_paths()[i].vertices

            if len(data) > len(vertices_for_fit):
                vertices_for_fit = data

        x_vertices = vertices_for_fit[:, 0]
        y_vertices = vertices_for_fit[:, 1]

        ########################################################################
        # Obtain the data for the circle fit. Only points that form the border
        # of the drop at 300 kg/m**3 are used and only the part close to the
        # spreading radius. Also, points that determine the lower border of the
        # drop along the surface are discarded.
        ellipse = []
        for i in range(int(0.5 * len(x_vertices))):
            if y_vertices[i] > 0.10:
                ellipse.append((x_vertices[i], y_vertices[i]))

        ########################################################################
        # Fit a circle to the data. x_spread_radius_fit and y_spread_radius_fit
        # are later used to plot the data into the figure.
        a_points = np.array(ellipse)
        lsc_data_ellipse = odr.Data(np.row_stack([a_points[:, 0], a_points[:, 1]]), y=1)
        lsc_model_spreading = odr.Model(f_3,
                                        implicit=True,
                                        estimate=calc_estimate,
                                        fjacd=jacd,
                                        fjacb=jacb)

        lsc_odr_spreading = odr.ODR(lsc_data_ellipse, lsc_model_spreading)
        lsc_odr_spreading.set_job(deriv=3)
        lsc_out_spreading = lsc_odr_spreading.run()
        x_spread_radius_fit = a_points[:, 0]
        y_spread_radius_fit = a_points[:, 1]

        x0, y0, r_contact = lsc_out_spreading.beta
        Ri = calc_radius(x_spread_radius_fit, y_spread_radius_fit, x0, y0)
        r_contact_def = np.mean(Ri)

        if y0 - r_contact_def > 0.0:
            x_intersect = 0.0
            slope_temp = 0.0
            contact_angle = 0.0
            slope1 = slope_temp
        else:
            x_intersect = np.sqrt(r_contact_def**2 - (0.0 - y0)**2) + x0
            slope_temp = (0.0 - y0) / (x_intersect - x0)
            angle_temp = np.arctan(slope_temp)
            contact_angle = 90.0 - np.rad2deg(angle_temp)
            slope1 = np.tan(np.deg2rad(180.0 - contact_angle))

        all_data.append(x_intersect)
        all_data.append(contact_angle)

        density_label = 'Density ρ, [kg/m' + chr(0x00B3) + ']'

        cbar = plt.colorbar(
            cntr,
            label=density_label
        )

        plt.clabel(
            cntrf,
            fontsize=11,
            font='Open Sans',
            fmt='%3.0f'
        )

        fig = plt.gcf()
        ax = fig.gca()

        angstrom_label = "r, [Å]"
        ax.set_xlabel(angstrom_label)
        ax.set_ylabel(angstrom_label)

        plt.text(
            50, 50,
            str((first_timestep + iframe * dt) / 1000) + " ps",
            fontsize=20,
            fontproperties='Open Sans'
        )

        plt.ylim(0, diameter)
        plt.xlim(0, diameter)

        plt.savefig(
            './drop_profile/Density-only-' +
            str(first_timestep + iframe * dt).zfill(7) +
            'fs.png'
        )

        plt.plot(a_points[:, 0], a_points[:, 1], lw=2, color='w')

        spreading_radius = plt.Circle(
            (x0, y0), r_contact_def,
            alpha=0.8,
            facecolor='none',
            edgecolor='orange',
            linewidth=2.0
            )

        ax.add_artist(spreading_radius)

        xpnts = np.linspace(x_intersect - 20.0, x_intersect + 20.0, 100)
        ypnts = slope1 * (xpnts - x_intersect)
        plt.plot(xpnts, ypnts, color='black', lw=2.0)

        # Formula for the ellipse
        # https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate#434482

        plt.savefig(
            './drop_profile/Density-with-CircleSlope' +
            str(first_timestep + iframe * dt).zfill(7) +
            'fs.png'
        )

        plt.close(fig)

    except:

        # If the above fails insert -99999.0 to trigger a case when this time step
        # should not be written to file containing all data.
        all_data.append(-99999.0)
        all_data.append(0.0)

        pass

    # And the same for the lowest molecular layer using get_spread_radius_map
    """
    z_height defines the thickness of the molecular layer: z_height * dz.
    """
    # spread_radius_xyz are for the contour plots.

    size_x = int(np.ceil(diameter / dx)) + 2
    size_y = size_x

    spread_radius_x = []
    spread_radius_y = []
    spread_radius_density = []

    for iy in range(-size_y, size_y):
        for ix in range(-size_x, size_x):
            volume = dx * dy * dz * z_height * 1.0e-30
            density = 0.0
            for iz in range(z_height):
                density += (xyz_distribution[ix, iy, iz]) / \
                           volume * m_molecule / 1000.0

            spread_radius_x.append(ix * dx)
            spread_radius_y.append(iy * dy)
            spread_radius_density.append(density)

    """
    From the data obtained in the last 2 loops, the contour plot, and the corresponding
    density levels used as colorplot labels will be plotted. A density of 400 kg/m**3 is
    used as outer limit defining the drop. To this limit the spreading radius will be fitted.
    """
    spread_radius_x = np.array(spread_radius_x)
    spread_radius_y = np.array(spread_radius_y)
    spread_radius_density = np.array(spread_radius_density)

    try:

        xi, yi = np.linspace(spread_radius_x.min(), spread_radius_x.max(), 1000), \
                 np.linspace(spread_radius_y.min(), spread_radius_y.max(), 1000)
        xi, yi = np.meshgrid(xi, yi)

        zi = scipy.interpolate.griddata(
            (spread_radius_x, spread_radius_y),
            spread_radius_density,
            (xi, yi),
            method='cubic'
        )

        for k in range(len(zi[1])):
            for j in range(len(zi[0])):
                if zi[j, k] < 0.0:
                    zi[j, k] = 0.0

        spread_radius_cntr_line = plt.contour(
            xi, yi, zi,
            levels[1:-1],
            colors='k',
            fontproperties='Open Sans'
        )

        # Get the data points needed to fit the circle.
        vertices_for_fit = []
        for i in range(len(spread_radius_cntr_line.collections[2].get_paths())):
            data = spread_radius_cntr_line.collections[2].get_paths()[i].vertices
            if len(data) > len(vertices_for_fit):
                vertices_for_fit = data

        if len(vertices_for_fit) == 0:
            all_data.append(0)
            all_data.append(0)
            all_data.append(0)
            return all_data

        x_spread_radius_fit = vertices_for_fit[:, 0]
        y_spread_radius_fit = vertices_for_fit[:, 1]

        lsc_data_spreading = odr.Data(
            np.row_stack(
                [x_spread_radius_fit, y_spread_radius_fit]
            ),
            y=1
        )

        lsc_model_spreading = odr.Model(f_3,
                                        implicit=True,
                                        estimate=calc_estimate,
                                        fjacd=jacd,
                                        fjacb=jacb
                                        )

        lsc_odr_spreading = odr.ODR(
            lsc_data_spreading,
            lsc_model_spreading
        )

        lsc_odr_spreading.set_job(deriv=3)
        lsc_out_spreading = lsc_odr_spreading.run()

        x_spread, y_spread, R_spread = lsc_out_spreading.beta
        Ri = calc_radius(x_spread_radius_fit, y_spread_radius_fit, x_spread, y_spread)
        r_spread_def = np.mean(Ri)

        all_data.append(x_spread)
        all_data.append(y_spread)
        all_data.append(r_spread_def)

        plt.clabel(
            spread_radius_cntr_line,
            fontsize=11,
            font='Open Sans',
            fmt='%3.0f'
        )

        spread_radius_cntr = plt.contourf(
            xi,
            yi,
            zi,
            levels[0:-1],
            vmin=levels[0], vmax=levels[-1],
            extent=[
                spread_radius_x.min(), spread_radius_x.max(),
                spread_radius_y.min(), spread_radius_y.max()]
        )

        spread_radius_cbar = plt.colorbar(
            spread_radius_cntr,
            label=density_label
        )

        spreading_fig = plt.gcf()
        ax = spreading_fig.gca()
        ax.set_xlabel(angstrom_label)
        ax.set_ylabel(angstrom_label)

        plt.text(
            50, 50,
            str(first_timestep / 1000) + " ps",
            fontsize=20,
            fontproperties='Open Sans'
        )

        plt.ylim(-diameter, diameter)
        plt.xlim(-diameter, diameter)

        plt.savefig(
            './drop_from_top/SpreadingDensityOnly_' +
            str(first_timestep + iframe * dt).zfill(7) +
            'fs.png')

        spreading_radius = plt.Circle(
            (x_spread, y_spread), r_spread_def,
            alpha=0.4,
            facecolor='none',
            edgecolor='black',
            linewidth=2.0
        )

        plt.plot(vertices_for_fit[:, 0], vertices_for_fit[:, 1], lw=2, color='w')

        ax.add_artist(
            spreading_radius
        )

        plt.savefig(
            './drop_from_top/SpreadingDensityWithCircle_' +
            str(first_timestep + iframe * dt).zfill(7) +
            'fs.png'
        )

        plt.close(spreading_fig)

    except:

        all_data.append(0.0)
        all_data.append(0.0)
        all_data.append(0.0)

        pass

    return all_data


def vel_dens_distribution(rho_z_mol_distrib,
                          rho_vel_distrib,
                          z_vel_distrib,
                          vel_abs_rhoz,
                          levels,
                          iframe, first_timestep, dt,
                          diameter,
                          dz, drho):

    from get_initial_data import cart2pol

    # avg_frames = np.floor(nFrames/navg)
    # Take average over navg frames. Discard any remaining frames.

    size_rho = int(np.ceil(diameter / drho)) + 2
    size_z = size_rho
    vel_rho = []
    vel_z = []
    vel_abs = []
    arrow_rho = []
    arrow_z = []

    # First, get the contour data for the velocity contour plot.

    norm = 0.0
    for iz in range(size_z):
        for irho in range(size_rho):
            vel_rho.append(irho * drho)
            vel_z.append(iz * dz)
            if rho_z_mol_distrib[irho, iz] == 0.0:
                vel_abs.append(0.0)
                arrow_rho.append(0.0)
                arrow_z.append(0.0)
            else:
                vel_abs.append(np.abs(vel_abs_rhoz[irho, iz]) / rho_z_mol_distrib[irho, iz])

                norm_temp = np.sqrt((rho_vel_distrib[irho, iz] / rho_z_mol_distrib[irho, iz])**2 +
                               (z_vel_distrib[irho, iz] / rho_z_mol_distrib[irho, iz])**2)
                if norm_temp > norm:
                    norm = norm_temp
                arrow_rho.append(rho_vel_distrib[irho, iz] / rho_z_mol_distrib[irho, iz])
                arrow_z.append(z_vel_distrib[irho, iz] / rho_z_mol_distrib[irho, iz])

    vel_rho = np.array(vel_rho)
    vel_z = np.array(vel_z)
    vel_abs = np.array(vel_abs)

    try:

        arrow_rho /= norm
        arrow_z /= norm
        arrow_rho = np.array(arrow_rho)
        arrow_z = np.array(arrow_z)

        if drho > dz:
            divisor = drho * 1.05
        else:
            divisor = dz * 1.05

        factor = vel_abs.max() / divisor
        # factor = np.ceil(vel_abs.max() / divisor)

        arrow_rho *= (vel_abs / factor)
        arrow_z *= (vel_abs / factor)

        xi, yi = np.linspace(vel_rho.min(), vel_rho.max(), 1000), \
                 np.linspace(vel_z.min(), vel_z.max(), 1000)
        xi, yi = np.meshgrid(xi, yi)

        zi = scipy.interpolate.griddata((vel_rho, vel_z), vel_abs, (xi, yi), method='cubic')

        for k in range(len(zi[1])):
            for j in range(len(zi[0])):
                if zi[j, k] < 0.0:
                    zi[j, k] = 0.0

        plt.figure()

        axes = plt.gca()

        axes.set_xlim([0, diameter + drho])
        axes.set_ylim([0, diameter + dz])

        cntrf = plt.contour(
            xi, yi, zi,
            levels[1:-1],
            colors='k',
            fontproperties='Open Sans'
        )

        plt.clabel(
            cntrf,
            fontsize=11,
            font='Open Sans',
            fmt='%3.0f'
        )

        cntr = plt.contourf(
            xi, yi, zi,
            levels[0:-1],
            vmin=levels[0], vmax=levels[-1]+100,
            origin='lower',
            extent=[
                vel_rho.min(), vel_rho.max(),
                vel_z.min(), vel_z.max()],
        )

        cbar = plt.colorbar(
            cntr,
            label='Velocity v, [m/s]'
        )

        fig = plt.gcf()
        ax = fig.gca()
        angstrom_label = "r, [Å]"
        ax.set_xlabel(angstrom_label)
        ax.set_ylabel(angstrom_label)

        plt.text(
            50, 50,
            str((first_timestep + iframe * dt) / 1000) + " ps",
            fontsize=20,
            fontproperties='Open Sans'
        )

        # Good examples for using quiver
        # http://stackoverflow.com/questions/25342072/computing-and-drawing-vector-fields#25343170
        # https://stackoverflow.com/questions/35047106/how-do-i-set-limits-on-ticks-colors-and-labels-for-colorbar-contourf-matplotli
        plt.quiver(vel_rho + drho / 2, vel_z + dz / 2, arrow_rho, arrow_z,
                   angles='xy',
                   scale_units='xy',
                   scale=1.0,
                   headlength=7,
                   width=0.004)

        plt.savefig(
            './velocity_distribution/VelocityDistribution-' +
            str(first_timestep + iframe * dt).zfill(7) +
            'fs.png'
        )

        plt.close(fig)

    except:
        pass
