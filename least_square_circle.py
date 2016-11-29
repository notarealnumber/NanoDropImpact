import numpy as np
from numpy.linalg import eig, inv
from scipy import optimize, odr
from matplotlib import pyplot as plt, cm, colors
import scipy.constants as constants

"""The functions and code snippets for the circle fitting have been taken from the Scipy cookbook:
https://scipy.github.io/old-wiki/pages/Cookbook/Least_Squares_Circle#Using_scipy.odr """

def calc_R(x, y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x - xc)**2 + (y - yc)**2)


def f_1(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


def f_3(beta, x):
    """
    Implicit definition of the circle
    """
    return (x[0] - beta[0])**2 + (x[1] - beta[1])**2 - beta[2]**2


def f_ellipse(beta, x):
    """
    Canonical form of an elipse: x^2/a^2 + y^2/b^2 = 1.0
    """
    return (x[0] / beta[0])**2 + (x[1] / beta[1])**2 - 1.0


def jacb(beta, x):
    """ Jacobian function with respect to the parameters beta.
    return df_3b/dbeta
    """
    xc, yc, r = beta
    xi, yi    = x

    df_db    = np.empty((beta.size, x.shape[1]))
    df_db[0] =  2*(xc-xi)                     # d_f/dxc
    df_db[1] =  2*(yc-yi)                     # d_f/dyc
    df_db[2] = -2*r                           # d_f/dr

    return df_db


def jacd(beta, x):
    """ Jacobian function with respect to the input x.
    return df_3b/dx
    """
    xc, yc, r = beta
    xi, yi    = x

    df_dx    = np.empty_like(x)
    df_dx[0] =  2*(xi-xc)                     # d_f/dxi
    df_dx[1] =  2*(yi-yc)                     # d_f/dyi

    return df_dx


def calc_estimate(data):
    """ Return a first estimation on the parameter from the data  """
    xc0, yc0 = np.mean(data.x, axis=1)
    r0 = np.sqrt((data.x[0] - xc0)**2 + (data.x[1] - yc0)**2).mean()
    return xc0, yc0, r0


def dummy_est():

    return 0.0, 0.0, 60.0


def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
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

# def fitEllipse(x, y):
#     """
#     http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html#
#     """
#     x = x[:, np.newaxis]
#     y = y[:, np.newaxis]
#     D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
#     S = np.dot(D.T, D)
#     C = np.zeros([6, 6])
#     C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
#     E, V = eig(np.dot(inv(S), C))
#     n = np.argmax(np.abs(E))
#     a = V[:, n]
#     return a
#
#
# def ellipse_center(veca):
#     b, c, d, f, g, a = veca[1]/2, veca[2], veca[3]/2, veca[4]/2, veca[5], veca[0]
#     num = b**2 - a*c
#     x0 = (c*d - b*f)/num
#     y0 = (a*f - b*d)/num
#     return np.array([x0, y0])
#
#
# def ellipse_angle_of_rotation2(veca):
#     """
#     Calculates the angle of rotation of an ellipsis.
#     """
#     b, c, d, f, g, a = veca[1]/2, veca[2], veca[3]/2, veca[4]/2, veca[5], veca[0]
#     if b == 0:
#         if a > c:
#             return 0
#         else:
#             return np.pi/2
#     else:
#         if a > c:
#             return np.arctan(2*b/(a-c))/2
#         else:
#             return np.pi/2 + np.arctan(2*b/(a-c))/2
#
#
# def ellipse_axis_length(veca):
#     """
#     Calculates the axis length of an ellipsis defined by a vector a.
#     Args:
#         a (float): vector that describes the ellipsis.
#     Returns:
#         res1, res2 (float): the length of the axis.
#     """
#
#     a, b, c, d, f, g = veca[0], veca[1]/2, veca[2], veca[3]/2, veca[4]/2, veca[5]
#
#     up = 2 * (a * f**2 + c*d**2 + g*b**2 - 2 * b*d*f - a*c*g)
#     # From mathworld.wolfram.com/Ellipse.html, equations 21 and 22 give the semi-major (21)
#     # and semi-minor (22) axis. Denominator is constructed by three parts:
#     part1 = b**2 - a*c
#     part2 = np.sqrt((a - c)**2 + 4 * b**2)
#     part3 = a + c
#     print(" ")
#     print(part1, part2, part3)
#
#     # For the semi-major axis the denominator is given as
#     denom1 = part1 * (part2 - part3)
#     # For the semi-minor axis the denominator is given as
#     denom2 = part1 * (-part2 - part3)
#     # down1 = (b*b - a*c) * ((c-a) * np.sqrt(1 + 4*b*b / ((a-c) * (a-c))) - (c+a))
#     # down2 = (b*b - a*c) * ((a-c) * np.sqrt(1 + 4*b*b / ((a-c) * (a-c))) - (c+a))
#
#     print(b, c, d, f, g, a)
#     # print(up, down1, down2)
#     print(up, denom1, denom2)
#     # print(up/down1)
#     print(up/denom1)
#     # print(up/down2)
#     print(up/denom2)
#     res1 = np.sqrt(np.abs(up/denom1))
#     res2 = np.sqrt(np.abs(up/denom2))
#
#     return np.array([res1, res2])


def lsc(x, y):
    # coordinates of the barycenter
    # x_m = np.mean(x)
    # y_m = np.mean(y)
    center_estimate = 0.0, 0.0 #x_m, y_m
    center, ier = optimize.leastsq(f_1, center_estimate, args=(x, y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu


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


# def plot_ellipsis(axes, center, phi):