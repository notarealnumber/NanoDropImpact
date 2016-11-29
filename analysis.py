import scipy.interpolate
from least_square_circle import *
import numpy as np
from scipy import odr


def spread_radius(rho_z_distribution,
                  xyz_distribution,
                  levels,
                  m_molecule,
                  iframe, first_timestep, dt,
                  xmin, xmax,
                  ymin, ymax,
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
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :param dx:
    :param dy:
    :param dz:
    :param drho:
    :return:
    """

    z_height = 2

    size_rho = int(np.ceil(100 / drho))
    size_z = int(np.ceil(80 / dz))
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

    # https://www.math.ucdavis.edu/~kouba/CalcOneDIRECTORY/implicitdiffdirectory/ImplicitDiff.html
    # www.derivative-calculator.net/

    rho_rho = np.array(rho_rho)
    rho_z = np.array(rho_z)
    rho_density = np.array(rho_density)

    xi, yi = np.linspace(rho_rho.min(), rho_rho.max(), 1000), \
        np.linspace(rho_z.min(), rho_z.max(), 1000)
    xi, yi = np.meshgrid(xi, yi)

    zi = scipy.interpolate.griddata((rho_rho, rho_z), rho_density, (xi, yi), method='cubic')

    for k in range(len(zi[1])):
        for j in range(len(zi[0])):
            if zi[j, k] < 0.0:
                zi[j, k] = 0.0

    # profile_fig = plt.figure(1)

    cntrf = plt.contour(
        xi, yi, zi,
        levels[1:-1],
        colors='k',
        fontproperties='Open Sans'
    )

    # print(cntrf.collections[2].get_paths()[1])
    v1 = cntrf.collections[2].get_paths()[0].vertices
    # v2 = cntrf.collections[2].get_paths()[1].vertices
    v = v1 #np.vstack([v1, v2])
    xcs = v[:, 0]
    ycs = v[:, 1]

    plt.clabel(
        cntrf,
        fontsize=11,
        font='Open Sans',
        fmt='%3.0f'
    )

    cntr = plt.contourf(
        xi,
        yi,
        zi,
        levels,
        vmin=rho_density.min(), vmax=rho_density.max(),
        origin='lower',
        extent=[
            rho_rho.min(), rho_rho.max(),
            rho_z.min(), rho_z.max()],
    )

    cbar = plt.colorbar(
        cntr,
        label=r'Density $\rho$ [kg/m$^{\text{3}}$]'
    )

    lsc_dataX = odr.Data(np.row_stack([xcs, ycs]), y=1)
    lsc_modelX = odr.Model(f_3, implicit=True, estimate=calc_estimate, fjacd=jacd, fjacb=jacb)
    lsc_odrX = odr.ODR(lsc_dataX, lsc_modelX)
    lsc_odrX.set_job(deriv=3)
    lsc_outX = lsc_odrX.run()

    xx_3, yx_3, Rx_3 = lsc_outX.beta
    Ri_3 = calc_R(xcs, ycs, xx_3, yx_3)
    Rx_def = np.mean(Ri_3)
    # residue = sum((Ri_3 - R_3)**2)

    all_data.append(xx_3)
    all_data.append(yx_3)
    all_data.append(Rx_3)

    fig = plt.gcf()
    ax = fig.gca()
    ax.set_xlabel(
        r"r, \AA"
    )

    ax.set_ylabel(
        r"z, \AA"
    )

    ax.set_xticklabels(
        ax.get_xticks(),
        fontproperties='Open Sans'
    )

    ax.set_yticklabels(
        ax.get_yticks(),
        fontproperties='Open Sans'
    )

    cbar.ax.set_yticklabels(
        levels,
        fontproperties='Open Sans'
    )

    plt.text(
        50, 50,
        str((first_timestep + iframe * dt) / 1000) + " ps",
        fontsize=20,
        fontproperties='Open Sans'
    )

    plt.savefig(
        './drop_profile/Density-only-' +
        str(first_timestep + iframe * dt).zfill(7) +
        'fs.png'
    )

    spread_radius_side = plt.Circle(
        (xx_3, yx_3), Rx_def,
        alpha=0.4,
        facecolor='none',
        edgecolor='black',
        linewidth=2.0
    )

    ax.add_artist(
        spread_radius_side
    )

    if yx_3 == 0.0:
        spread_rad_surf = Rx_def
        deriv = 0.0
    elif yx_3 > 0.0:
        spread_rad_surf = np.sqrt(Rx_def**2 - yx_3**2) + xx_3
        deriv = (spread_rad_surf - xx_3) / np.sqrt(Rx_def**2 - (spread_rad_surf - xx_3)**2)
    elif yx_3 < 0.0:
        spread_rad_surf = np.sqrt(Rx_def**2 - yx_3**2) + xx_3
        deriv = (-1) * (spread_rad_surf - xx_3) / np.sqrt(Rx_def**2 - (spread_rad_surf - xx_3)**2)

    contact_angle = np.absolute(np.rad2deg(np.arctan(deriv)))

    plt.ylim(0, rho_z.max())
    plt.xlim(0, rho_rho.max())

    x = np.linspace(Rx_def - 40.0, Rx_def, 25)
    slope = deriv * x - spread_rad_surf * deriv
    plt.plot(x, slope, alpha=0.4, color='yellow', linewidth=2.0)

    plt.savefig(
        './drop_profile/Density-with-CircleSlope' +
        str(first_timestep + iframe * dt).zfill(7) +
        'fs.png'
    )

    plt.close(fig)
    # plt.close(profile_fig)

    # print(spread_rad_surf, deriv, contact_angle, np.absolute(contact_angle))
    all_data.append(spread_rad_surf)
    all_data.append(contact_angle)

    # And the same for the disks using get_spread_radius_map

    for iz in range(0, z_height * 3, z_height):
        """
        z_height defines how many layers of thickness dz will be merged.
        Only 4 layers of thickness z_height * dz will be evaluated.
        """
        # spread_radius_xyz are for the contour plots.
        spread_radius_x = []
        spread_radius_y = []
        spread_radius_density = []

        for iy in range(2 * ymin, 2 * ymax):
            for ix in range(2 * xmin, 2 * xmax):
                volume = dx * dy * dz * z_height * 1.0e-30
                density = 0.0
                for z in range(z_height):
                    density += xyz_distribution[ix, iy, iz + z] / volume * m_molecule / 1000.0

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

        # print(spread_radius_density.max(), "ff")

        if spread_radius_density.max() < 200.0:
            break

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

        # levels = ()
        # lvl_rnge = int(np.ceil(spread_radius_density.max() / 100) * 100) + 100
        # for lvl in range(0, lvl_rnge, 100):
        #     levels = levels + (lvl,)

        # spread_radius_fig = plt.figure(iz)

        spread_radius_cntr_line = plt.contour(
            xi, yi, zi,
            levels[1:-1],
            colors='k',
            fontproperties='Open Sans'
        )

        # Get the data points needed to fit the circle.
        # 200 kg/m**3 should be used as limit.
        pts_for_fit = spread_radius_cntr_line.collections[2].get_paths()[0]
        vertices_for_fit = pts_for_fit.vertices
        x_spread_radius_fit = vertices_for_fit[:, 0]
        y_spread_radius_fit = vertices_for_fit[:, 1]

        lsc_data_spreading = odr.Data(np.row_stack([x_spread_radius_fit, y_spread_radius_fit]), y=1)
        lsc_model_spreading = odr.Model(f_3, implicit=True, estimate=calc_estimate, fjacd=jacd, fjacb=jacb)
        lsc_odr_spreading = odr.ODR(lsc_data_spreading, lsc_model_spreading)
        lsc_odr_spreading.set_job(deriv=3)
        lsc_out_spreading = lsc_odr_spreading.run()

        x_spread, y_spread, R_spread = lsc_out_spreading.beta
        Ri = calc_R(x_spread_radius_fit, y_spread_radius_fit, x_spread, y_spread)
        r_spread_def = np.mean(Ri)
        # r_h_per_frame.append([R_def, iz * dz + dz * z_height / 2.0])
        # residue = sum((Ri_3 - R_3)**2)
        # print(x_spread, y_spread, r_spread_def, "eee")
        all_data.append(int(iz / z_height))
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
            levels,
            extent=[
                spread_radius_x.min(), spread_radius_x.max(),
                spread_radius_y.min(), spread_radius_y.max()]
        )

        spread_radius_cbar = plt.colorbar(
            spread_radius_cntr,
            label=r'Density $\rho$ [kg/m$^3$]'
        )

        spreading_fig = plt.gcf()
        ax = spreading_fig.gca()
        ax.set_xlabel(
            r"r, \AA"
        )
        ax.set_ylabel(
            r"z, \AA"
        )
        ax.set_xticklabels(
            ax.get_xticks(),
            fontproperties='Open Sans'
        )

        ax.set_yticklabels(
            ax.get_yticks(),
            fontproperties='Open Sans'
        )

        spread_radius_cbar.ax.set_yticklabels(
            levels,
            fontproperties='Open Sans'
        )

        plt.text(
            50, 50,
            str(first_timestep / 1000) + " ps",
            fontsize=20,
            fontproperties='Open Sans'
        )

        plt.ylim(-(r_spread_def + 20.0), r_spread_def + 20.0)
        plt.xlim(-(r_spread_def + 20.0), r_spread_def + 20.0)

        plt.savefig(
            './drop_from_top/SpreadingDensityOnly_Layer' +
            str(int(iz / z_height)) + "_" +
            str(first_timestep + iframe * dt).zfill(7) +
            'fs.png')

        spreading_radius = plt.Circle(
            (x_spread, y_spread), r_spread_def,
            alpha=0.4,
            facecolor='none',
            edgecolor='black',
            linewidth=2.0
        )

        ax.add_artist(
            spreading_radius
        )

        plt.savefig(
            './drop_from_top/SpreadingDensityWithCircle_Layer' +
            str(int(iz / z_height)) + "_" +
            str(first_timestep + iframe * dt).zfill(7) +
            'fs.png'
        )

        plt.close(spreading_fig)

    return all_data


def vel_dens_distribution(xyz_mol_distrib,
                          rho_z_mol_distrib,
                          rho_vel_distrib,
                          z_vel_distrib,
                          vel_abs_rhoz,
                          vel_abs_xyz,
                          iframe, first_timestep, dt,
                          dx, dy, dz, drho):

    from get_initial_data import cart2pol

    # avg_frames = np.floor(nFrames/navg)
    # Take average over navg frames. Discard any remaining frames.

    size_rho = int(np.ceil(100 / drho))
    size_z = int(np.ceil(80 / dz))
    vel_rho = []
    vel_z = []
    vel_abs = []
    arrow_rho = []
    arrow_z = []
    drhozsq = np.sqrt(drho**2 + dz**2)

    # First, get the contour data for the velocity contour plot.
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

                norm = np.sqrt((rho_vel_distrib[irho, iz] / rho_z_mol_distrib[irho, iz])**2 +
                               (z_vel_distrib[irho, iz] / rho_z_mol_distrib[irho, iz])**2)
                arrow_rho.append(rho_vel_distrib[irho, iz] / rho_z_mol_distrib[irho, iz] / norm)
                arrow_z.append(z_vel_distrib[irho, iz] / rho_z_mol_distrib[irho, iz] / norm)

    vel_rho = np.array(vel_rho)
    vel_z = np.array(vel_z)
    vel_abs = np.array(vel_abs)
    arrow_rho = np.array(arrow_rho)
    arrow_z = np.array(arrow_z)

    if drho > dz:
        divisor = drho
    elif drho < dz:
        divisor = dz
    else:
        divisor = dz

    factor = np.ceil(vel_abs.max() / divisor)

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

    # profile_fig = plt.figure(1)
    dtick = int(np.ceil(np.floor(zi.max()/10) / 10) * 10)
    maxtick = int(np.ceil(zi.max() / dtick) * dtick + 2 * dtick)

    plt.figure()
    # print(rho_z_mol_distrib[0])
    # exit()
    # plt.show(arrow_plot)
    # exit()

    # Good examples for using quiver
    # http://stackoverflow.com/questions/25342072/computing-and-drawing-vector-fields#25343170
    # https://stackoverflow.com/questions/35047106/how-do-i-set-limits-on-ticks-colors-and-labels-for-colorbar-contourf-matplotli

    levels = ()
    for lvl in range(0, maxtick, dtick):
        levels += (lvl,)

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
        vmin=vel_abs.min(), vmax=vel_abs.max(),
        origin='lower',
        extent=[
            vel_rho.min(), vel_rho.max(),
            vel_z.min(), vel_z.max()],
    )

    cbar = plt.colorbar(
        cntr,
        label=r'Velocity \textit{v} [m/s]'
    )

    fig = plt.gcf()
    ax = fig.gca()
    ax.set_xlabel(
        r"r, \AA"
    )

    ax.set_ylabel(
        r"z, \AA"
    )

    # ax.set_xticklabels(
    #     ax.get_xticks(),
    #     fontproperties='Open Sans'
    # )
    #
    # ax.set_yticklabels(
    #     ax.get_yticks(),
    #     fontproperties='Open Sans'
    # )

    plt.text(
        50, 50,
        str((first_timestep + iframe * dt) / 1000) + " ps",
        fontsize=20,
        fontproperties='Open Sans'
    )

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

    # plt.show()

    # plt.ylim(0, vel_z.max())
    # plt.xlim(0, vel_rho.max())

    plt.close(fig)

    # Second, prepare the data for the velocity vectors that will be layed over the
    # contour plot.
