import numpy as np
from numpy.linalg import eig, inv
from scipy import optimize
from matplotlib import pyplot as plt
import scipy.constants as constants

"""The functions and code snippets for the circle fitting have been taken from the Scipy cookbook:
https://scipy.github.io/old-wiki/pages/Cookbook/Least_Squares_Circle#Using_scipy.odr """


def calc_radius(x, y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x - xc)**2 + (y - yc)**2)


def f_1(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    radius_i = calc_radius(x, y, *c)
    return radius_i - radius_i.mean()


def f_3(beta, x):
    """
    Implicit definition of the circle
    """
    return (x[0] - beta[0])**2 + (x[1] - beta[1])**2 - beta[2]**2


# Yet another way to fit an ellipse:
# http://scipython.com/book/chapter-8-scipy/examples/non-linear-fitting-to-an-ellipse/
def f_ellipse2(theta, p):
    a, e = p
    return a * (1 - e**2)/(1 - e * np.cos(theta))


def residuals(p, r, theta):
    """ Return the observed - calculated residuals using f(theta, p). """
    return r - f_ellipse2(theta, p)


def jac(p, r, theta):
    """ Calculate and return the Jacobian of residuals. """
    a, e = p
    da = (1 - e**2)/(1 - e*np.cos(theta))
    de = (-2*a*e*(1-e*np.cos(theta)) +
          a*(1-e**2)*np.cos(theta))/(1 - e*np.cos(theta))**2
    return -da,  -de
    return np.array((-da, -de)).T


def jacb(beta, x):
    """ Jacobian function with respect to the parameters beta.
    return df_3b/dbeta
    """
    xc, yc, r = beta
    xi, yi = x

    df_db = np.empty((beta.size, x.shape[1]))
    df_db[0] = 2 * (xc - xi)                     # d_f/dxc
    df_db[1] = 2 * (yc - yi)                     # d_f/dyc
    df_db[2] = -2 * r                           # d_f/dr

    return df_db


def jacd(beta, x):
    """ Jacobian function with respect to the input x.
    return df_3b/dx
    """
    xc, yc, r = beta
    xi, yi = x

    df_dx = np.empty_like(x)
    df_dx[0] = 2 * (xi - xc)  # d_f/dxi
    df_dx[1] = 2 * (yi - yc)  # d_f/dyi

    return df_dx


def calc_estimate(data):
    """ Return a first estimation on the parameter from the data  """
    xc0, yc0 = np.mean(data.x, axis=1)
    r0 = np.sqrt((data.x[0] - xc0)**2 + (data.x[1] - yc0)**2).mean()
    return xc0, yc0, r0


def f_ellipse_rotated(beta, x):
    """
    http://www.maa.org/external_archive/joma/Volume8/Kalman/General.html
    Canonical form of an ellipse:
    A * (x - h)**2 + B * (x - h)(y - k) + C * (y - k)**2 = 1.0
    where
    A = cos(phi)**2 / a**2 + sin(phi)**2 / b**2
    B = sin(2 * phi) * (1 / a**2 - 1 / b**2)
    C = sin(phi)**2 / a**2 + cos(phi)**2 / b**2

    Standard form: x^2/a^2 + y^2/b^2 = 1.0

    h = beta[0]
    k = beta[1]
    a = beta[2]
    b = beta[3]
    phi = beta[4]
    x = x[0]
    y = x[1]
    """
    h = beta[0]
    k = beta[1]
    a = beta[2]
    b = beta[3]
    phi = beta[4]

    A = (np.cos(phi)**2 / a**2 + np.sin(phi)**2 / b**2)
    B = np.sin(2.0 * phi) * (1.0 / a**2 - 1.0 / b**2)
    C = (np.sin(phi)**2 / a**2 + np.cos(phi)**2 / b**2)

    return A * (x[0] - h)**2 + B * (x[0] - h) * (x[1] - k) + C * (x[1] - k)**2 - 1.0


def jacb_ellipse_rotated(beta, x):
    """ Jacobian function with respect to the parameters beta.
    return df_3b/dbeta
    """
    h, k, a, b, phi = beta
    xi, yi = x

    cos_sin = (np.cos(phi)**2 / a**2 + np.sin(phi)**2 / b**2)
    sin_cos = (np.sin(phi)**2 / a**2 + np.cos(phi)**2 / b**2)
    a2_b2 = (1.0 / a**2 - 1.0 / b**2)
    b2_a2 = (1.0 / b**2 - 1.0 / a**2)

    df_db = np.empty((beta.size, x.shape[1]))

    df_db[0] = (-2.0) * (xi - h) * cos_sin - a2_b2 * (yi - k) * np.sin(2.0 * phi)  # df/dh
    df_db[1] = (-2.0) * (yi - k) * sin_cos - a2_b2 * (xi - h) * np.sin(2.0 * phi)  # df/dk
    df_db[2] = (-2.0 / a**3) * ((xi - h) * np.cos(phi) + (yi - k) * np.sin(phi))**2  # df/da
    df_db[3] = (-2.0 / b**3) * ((xi - h) * np.sin(phi) - (yi - k) * np.cos(phi))**2  # df/db
    df_db[4] = 2.0 * a2_b2 * (xi - h) * (yi - k) * np.cos(2.0 * phi) + \
               np.sin(2.0 * phi) * \
               (b2_a2 * (xi - h)**2 + a2_b2 * (yi - k)**2)  # d_f/dphi

    return df_db


def jacd_ellipse_rotated(beta, x):
    """ Jacobian function with respect to the input x.
    return df_3b/dx
    """
    h, k, a, b, phi = beta
    xi, yi = x

    cos_sin = (np.cos(phi)**2 / a**2 + np.sin(phi)**2 / b**2)
    sin_cos = (np.sin(phi)**2 / a**2 + np.cos(phi)**2 / b**2)
    a2_b2 = (1.0 / a**2 - 1.0 / b**2)

    df_dx = np.empty_like(x)
    df_dx[0] = a2_b2 * (yi - k) * np.sin(2.0 * phi) + 2.0 * (xi - h) * cos_sin  # d_f/dx
    df_dx[1] = a2_b2 * (xi - h) * np.sin(2.0 * phi) + 2.0 * (yi - k) * sin_cos  # d_f/dx

    return df_dx


def calc_estimate_ellipse_rotated(data):
    """ Return a first estimation on the parameter from the data  """

    xmax, ymax = np.max(data.x, axis=1)
    xmin, ymin = np.min(data.x, axis=1)

    h = (xmax - xmin) / 2.0 + xmin
    k = (ymax - ymin) / 2.0 + ymin
    a = xmax - h
    b = ymax - k
    phi = 0.1

    print("Initial guess: ", h, k, a, b, phi)

    return h, k, a, b, phi


def fit_ellipse2(beta, x):
    """
    http://mathworld.wolfram.com/Ellipse.html
    :param x:
    :param y:
    :return:
    """
    a = beta[0]
    b = beta[1]
    c = beta[2]
    d = beta[3]
    f = beta[4]
    g = beta[5]

    return a * x[0]**2 + 2.0 * b * x[0] * x[1] + c * x[1]**2 + \
        2.0 * d * x[0] + 2.0 * f * x[1] + g


def jacb_ellipse2(beta, x):
    """ Jacobian function with respect to the parameters beta.
    return df_ellipse/dbeta
    """
    xi, yi = x
    df_db = np.empty((beta.size, x.shape[1]))

    df_db[0] = xi**2
    df_db[1] = 2.0 * xi * yi
    df_db[2] = yi**2
    df_db[3] = 2.0 * xi
    df_db[4] = 2.0 * yi
    df_db[5] = 1

    return df_db


def jacd_ellipse2(beta, x):
    """ Jacobian function with respect to the parameters beta.
    return df_ellipse/dbeta
    """
    a, b, c, d, f, g = beta
    xi, yi = x
    df_dx = np.empty_like(x)

    df_dx[0] = 2.0 * (a * xi + b * yi + d)
    df_dx[1] = 2.0 * (b * xi + c * yi + f)

    return df_dx


def f_ellipse(beta, x):
    """
    Canonical form of an ellipse:
    (x - h)**2/a**2 + (y - k)**2/b**2 = 1.0

    Standard form: x^2/a^2 + y^2/b^2 = 1.0

    x = x[0]
    y = x[1]
    """
    h = beta[0]
    k = beta[1]
    a = beta[2]
    b = beta[3]

    return ((x[0] - h)**2 / a**2) + ((x[1] - k)**2 / b**2) - 1.0


def jacb_ellipse(beta, x):
    """ Jacobian function with respect to the parameters beta.
    return df_ellipse/dbeta
    """
    h, k, a, b = beta
    xi, yi = x

    df_db = np.empty((beta.size, x.shape[1]))

    df_db[0] = 2.0 * (h - xi) / a**2  # df/dh
    df_db[1] = 2.0 * (k - yi) / b**2  # df/dk
    df_db[2] = (-2.0) * (xi - h)**2 / a**3  # df/da
    df_db[3] = (-2.0) * (yi - k)**2 / b**3  # df/db

    return df_db


def jacd_ellipse(beta, x):
    """ Jacobian function with respect to the input x.
    return df_3b/dx
    """
    h, k, a, b = beta
    xi, yi = x

    df_dx = np.empty_like(x)
    df_dx[0] = 2.0 * (xi - h) / a**2  # d_f/dx
    df_dx[1] = 2.0 * (yi - k) / b**2  # d_f/dy

    return df_dx


def calc_estimate_ellipse(data):
    """ Return a first estimation on the parameter from the data  """

    xmax, ymax = np.max(data.x, axis=1)
    xmin, ymin = np.min(data.x, axis=1)

    h = (xmax - xmin) / 2.0 + xmin
    k = (ymax - ymin) / 2.0 + ymin
    a = xmax - h
    b = ymax - k

    print("Initial guess: ", h, k, a, b)

    return h, k, a, b


def dummy_est():

    return 0.0, 0.0, 60.0


def fit_ellipse(x, y):
    #     """
    #     http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html#
    #     """

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a


def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])


def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2


def lsc(x, y):
    # coordinates of the barycenter
    # x_m = np.mean(x)
    # y_m = np.mean(y)
    center_estimate = 0.0, 0.0 #x_m, y_m
    center, ier = optimize.leastsq(f_1, center_estimate, args=(x, y))
    xc, yc = center
    Ri = calc_radius(x, y, *center)
    R = Ri.mean()
    residue = np.sum((Ri - R)**2)
    return xc, yc, R, residue


def plot_data_circle(x, y, xc, yc, R):
    f = plt.figure(facecolor='white')  #figsize=(7, 5.4), dpi=72,
    plt.axis('equal')

    theta_fit = np.linspace(-constants.pi, constants.pi, 180)

    x_fit = xc + R*np.cos(theta_fit)
    y_fit = yc + R*np.sin(theta_fit)
    plt.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=2)
    plt.plot([xc], [yc], 'bD', mec='y', mew=1)
    plt.xlabel('x')
    plt.ylabel('y')
    # plot data
    plt.plot(x, y, 'r.', label='data', mew=1)

    plt.legend(loc='best',labelspacing=0.1 )
    plt.grid()
    plt.title('Least Squares Circle')
    plt.show()
