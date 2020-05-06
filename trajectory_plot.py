
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.collections as mcoll
import torch as torch
from matplotlib.patches import Ellipse

def gaussian(x, y, xmean, ymean, sigma):
    # gaussian to used as fit.
    return np.exp(-((x-xmean) ** 2 + (y-ymean) ** 2) / (2 * sigma ** 2))

def draw_lines(output,output_i,linestyle='-',alpha=1,darker=False,linewidth=2):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    loc = np.array(output[output_i,:,:,0:2])
    loc = np.transpose( loc, [1,2,0] )

    x = loc[:,0,:]
    y = loc[:,1,:]
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    max_range = max( y_max-y_min, x_max-x_min )
    xmin = (x_min+x_max)/2-max_range/2-0.1
    xmax = (x_min+x_max)/2+max_range/2+0.1
    ymin = (y_min+y_max)/2-max_range/2-0.1
    ymax = (y_min+y_max)/2+max_range/2+0.1

    cmaps = [ 'Purples', 'Greens', 'Blues', 'Oranges', 'Reds', 'Purples', 'Greens', 'Blues', 'Oranges', 'Reds'  ]
    cmaps = [ matplotlib.cm.get_cmap(cmap, 512) for cmap in cmaps ]
    cmaps = [ ListedColormap(cmap(np.linspace(0., 0.8, 256))) for cmap in cmaps ]
    if darker:
        cmaps = [ ListedColormap(cmap(np.linspace(0.2, 0.8, 256))) for cmap in cmaps ]

    for i in range(loc.shape[-1]):
        lc = colorline(loc[:,0,i], loc[:,1,i], cmap=cmaps[i],linestyle=linestyle,alpha=alpha,linewidth=linewidth)
    return xmin, ymin, xmax, ymax



def draw_lines_animation(output,linestyle='-',alpha=1,darker=False,linewidth=2, animationtype = 'default'):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    # animation for output  used to show how physical and computational errors propagate through system
    global xmin, xmax, ymin, ymax
    # output here is of form [perturbation,  particles, timestep,(x,y)]
    import matplotlib.pyplot as plt
    from matplotlib import animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # scaling of variables.
    loc_new = np.array(output)
    loc_new_x = loc_new[:, :, :, 0]
    loc_new_y = loc_new[:, :, :, 1]
    fig = plt.figure()
    x_min = np.min(loc_new_x[:,:,0:100])
    x_max = np.max(loc_new_x[:,:,0:100])
    y_min = np.min(loc_new_y[:,:,0:100])
    y_max = np.max(loc_new_y[:,:,0:100])
    max_range = max( y_max-y_min, x_max-x_min )
    xmin = (x_min+x_max)/2-max_range/2-0.1
    xmax = (x_min+x_max)/2+max_range/2+0.1
    ymin = (y_min+y_max)/2-max_range/2-0.1
    ymax = (y_min+y_max)/2+max_range/2+0.1
    # if x >= xmax - 1.00:
    #     p011.axes.set_xlim(x - xmax + 1.0, x + 1.0)
    #     p021.axes.set_xlim(x - xmax + 1.0, x + 1.0)
    #     p031.axes.set_xlim(x - xmax + 1.0, x + 1.0)
    #     p032.axes.set_xlim(x - xmax + 1.0, x + 1.0)
    # plots for animation
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin,ymax))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    line, = ax.plot([],[],lw = 1)
    lines = []
    cmaps = [ 'Purples', 'Greens', 'Blues', 'Oranges', 'Reds', 'Purples', 'Greens', 'Blues', 'Oranges', 'Reds'  ]
    cmaps = [ matplotlib.cm.get_cmap(cmap, 512) for cmap in cmaps ]
    cmaps = [ ListedColormap(cmap(np.linspace(0., 0.8, 256))) for cmap in cmaps ]
    if darker:
        cmaps = [ ListedColormap(cmap(np.linspace(0.2, 0.8, 256))) for cmap in cmaps ]
    for i in range(len(loc_new_x)):
        for j in range(len(loc_new_x[i])):
            colour = cmaps[j].colors[int(len(cmaps[j].colors)-1)]
            lobj = ax.plot([],[], lw =1, color = colour)[0]
            lines.append(lobj)

    def init():
        # initialise the lines to be plotted
        for line in lines:
            line.set_data([],[])
        return lines

    xdata, ydata = [], []

    def animate(i, xmax, xmin, ymax, ymin):
        # animation step
        xlist = []
        ylist = []
        if (i<=50):
            for j in range(len(loc_new_x)):
                for k in range(len(loc_new_x[j])):
                    x = loc_new_x[j][k][0:i]
                    y = loc_new_y[j][k][0:i]
                    xlist.append(x)
                    ylist.append(y)
            for lnum, line in enumerate(lines):
                line.set_data(xlist[lnum], ylist[lnum])
        else:
            for j in range(len(loc_new_x)):
                for k in range(len(loc_new_x[j])):
                    x = loc_new_x[j][k][i-50:i]
                    y = loc_new_y[j][k][i-50:i]
                    xlist.append(x)
                    ylist.append(y)
            if (np.any(xlist < xmin)) or (np.any(xlist > xmax)) or (np.any(ylist<ymin)) or (np.any(ylist>ymax)):
                x_min, x_max, y_min, y_max = np.amin(np.asarray(xlist)), np.amax(np.asarray(xlist)), np.amin(np.asarray(ylist)), np.amax(np.asarray(ylist))
                max_range = max(y_max - y_min, x_max - x_min)
                xmin = (x_min + x_max) / 2 - max_range / 2 - 0.4
                xmax = (x_min + x_max) / 2 + max_range / 2 + 0.4
                ymin = (y_min + y_max) / 2 - max_range / 2 - 0.4
                ymax = (y_min + y_max) / 2 + max_range / 2 + 0.4
                for lnum, line in enumerate(lines):
                    line.axes.set_xlim(xmin, xmax)
                    line.axes.set_ylim(ymin, ymax)
            for lnum, line in enumerate(lines):
                line.set_data(xlist[lnum], ylist[lnum])
        return lines


    anim = animation.FuncAnimation(fig, animate, init_func=init, frames = len(loc_new_x[0][0]),fargs= (xmax, xmin, ymax, ymin) ,interval = 10)
    plt.show()
    anim.save(animationtype + '.mp4', writer = writer)


def draw_lines_sigma(output,output_i,sigma_plot,ax, linestyle='-',alpha=1,  darker=False,linewidth=2, plot_ellipses= False):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    # plot the trajectories for sigma with ellipses of size sigma used to visualise the predictions of sigma we get out.
    loc = np.array(output[output_i,:,:,0:2])
    loc = np.transpose( loc, [1,2,0] )
    # scaling of variables.
    x = loc[:,0,:]
    y = loc[:,1,:]
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    max_range = max( y_max-y_min, x_max-x_min )
    xmin = (x_min+x_max)/2-max_range/2-0.1
    xmax = (x_min+x_max)/2+max_range/2+0.1
    ymin = (y_min+y_max)/2-max_range/2-0.1
    ymax = (y_min+y_max)/2+max_range/2+0.1

    cmaps = [ 'Purples', 'Greens', 'Blues', 'Oranges', 'Reds', 'Purples', 'Greens', 'Blues', 'Oranges', 'Reds'  ]
    cmaps = [ matplotlib.cm.get_cmap(cmap, 512) for cmap in cmaps ]
    cmaps = [ ListedColormap(cmap(np.linspace(0., 0.8, 256))) for cmap in cmaps ]
    if darker:
        cmaps = [ ListedColormap(cmap(np.linspace(0.2, 0.8, 256))) for cmap in cmaps ]
    # ensure we use the same colour for ellipses as for the particles, also use a small alpha to make it more transparent
    for i in range(loc.shape[-1]):
        lc = colorline(loc[:,0,i], loc[:,1,i], cmap=cmaps[i],linestyle=linestyle,alpha=alpha,linewidth=linewidth)
        if plot_ellipses:
            # isotropic therefore the ellipses become circles
            colour = cmaps[i].colors[int(len(cmaps[i].colors)/4)]
            positions = output[output_i,:,:,0:2]
            sigma_plot_pos = sigma_plot[output_i, :,:,0:2]

            ellipses = []
            # get the first timestep component of (x,y)
            ellipses.append(Ellipse((positions[i][0][0], positions[i][0][1]),
                                       width=sigma_plot_pos[i][0][0],
                                       height=sigma_plot_pos[i][0][0], angle=0.0, color = colour))
            # if Deltax^2+Deltay^2>4*(DeltaSigmax^2+DeltaSigma^2) then plot, else do not plot
            # keeps track of current plot value
            l = 0
            for k in range(len(positions[i]) - 1):
                deltar = np.linalg.norm(positions[i][k + 1] - positions[i][l])
                deltasigma = np.linalg.norm(sigma_plot_pos[i][l])
                if (deltar > 2 * deltasigma):
                    # check that it is far away from others
                    isfarapart = True
                    for m in range(len(positions)):
                        for n in range(len(positions[m])):
                            if (m != i):
                                deltar = np.linalg.norm(positions[m][n] - positions[i][k + 1])
                                deltasigma = np.linalg.norm(sigma_plot_pos[i][k + 1])
                                if (deltar < deltasigma):
                                    isfarapart = False
                    if isfarapart:
                        ellipses.append(Ellipse((positions[i][k + 1][0], positions[i][k + 1][1]),
                                                    width=sigma_plot_pos[i][k + 1][0],
                                                    height=sigma_plot_pos[i][k + 1][0], angle=0.0, color = colour))
                        # updates to new r0 : Deltar = r - r0:
                        l = k
            # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

            for e in ellipses:
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
    return xmin, ymin, xmax, ymax


def draw_lines_anisotropic(output,output_i,sigma_plot, vel_plot, ax, linestyle='-',alpha=1,  darker=False,linewidth=2, plot_ellipses= False):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    # plot the trajectories for sigma with ellipses of size sigma used to visualise the predictions of sigma we get out.
    # here we use anisotropic sigma case
    loc = np.array(output[output_i,:,:,0:2])
    loc = np.transpose( loc, [1,2,0] )

    x = loc[:,0,:]
    y = loc[:,1,:]
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    max_range = max( y_max-y_min, x_max-x_min )
    xmin = (x_min+x_max)/2-max_range/2-0.1
    xmax = (x_min+x_max)/2+max_range/2+0.1
    ymin = (y_min+y_max)/2-max_range/2-0.1
    ymax = (y_min+y_max)/2+max_range/2+0.1

    cmaps = [ 'Purples', 'Greens', 'Blues', 'Oranges', 'Reds', 'Purples', 'Greens', 'Blues', 'Oranges', 'Reds'  ]
    cmaps = [ matplotlib.cm.get_cmap(cmap, 512) for cmap in cmaps ]
    cmaps = [ ListedColormap(cmap(np.linspace(0., 0.8, 256))) for cmap in cmaps ]
    if darker:
        cmaps = [ ListedColormap(cmap(np.linspace(0.2, 0.8, 256))) for cmap in cmaps ]
    # ensure we use the same colour for ellipses as for the particles, also use a small alpha to make it more transparent
    for i in range(loc.shape[-1]):
        lc = colorline(loc[:,0,i], loc[:,1,i], cmap=cmaps[i],linestyle=linestyle,alpha=alpha,linewidth=linewidth)
        if plot_ellipses:
            colour = cmaps[i].colors[int(len(cmaps[i].colors) / 4)]
            positions = output[output_i, :, :, 0:2]
            sigma_plot_pos = sigma_plot[output_i, :, :, 0:2]
            indices_3 = torch.LongTensor([0])
            if vel_plot.is_cuda:
                indices_3 = indices_3.cuda()
            # plots the uncertainty ellipses for gaussian case.
            # iterate through each of the atoms
            # need to get the angles of the terms to be plotted:
            velnorm = vel_plot.norm(p=2, dim=3, keepdim=True)
            normalisedvel = vel_plot.div(velnorm.expand_as(vel_plot))
            normalisedvel[torch.isnan(normalisedvel)] = np.power(1 / 2, 1 / 2)
            # v||.x is just the first term of the tensor
            normalisedvelx = torch.index_select(normalisedvel, 3, indices_3)
            # angle of rotation is Theta = acos(v||.x) for normalised v|| and x (need angle in degrees not radians)
            angle = torch.acos(normalisedvelx).squeeze() * 180 / 3.14159
            ellipses = []
            ellipses.append(
                Ellipse((positions[i][0][0], positions[i][0][1]),
                        width=sigma_plot_pos[i][0][0],
                        height=sigma_plot_pos[i][0][1], angle=angle.tolist()[output_i][i][0], color = colour))
            # iterate through each of the atoms
            # if Deltax^2+Deltay^2>4*(DeltaSigmax^2+DeltaSigma^2) then plot, else do not plot
            # keeps track of current plot value
            l = 0
            for k in range(len(positions[i]) - 1):
                deltar = np.linalg.norm(positions[i][k + 1] - positions[i][l])
                deltasigma = np.linalg.norm(sigma_plot_pos[i][l])
                if (deltar > 2 * deltasigma):
                        # check that it is far away from others
                    isfarapart = True
                    for m in range(len(positions)):
                        for n in range(len(positions[m])):
                            if (m != i):
                                deltar = np.linalg.norm(positions[m][n] - positions[i][k + 1])
                                deltasigma = np.linalg.norm(sigma_plot_pos[i][k + 1])
                                if (deltar < deltasigma):
                                    isfarapart = False
                    if isfarapart:
                        ellipses.append(Ellipse(
                                (positions[i][k + 1][0], positions[i][k + 1][1]),
                                width=sigma_plot_pos[i][k + 1][0],
                                height=sigma_plot_pos[i][k + 1][0], angle=angle.tolist()[output_i][i][k + 1], color = colour))
                        # updates to new r0 : Deltar = r - r0:
                        l = k

            for e in ellipses:
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
    return xmin, ymin, xmax, ymax

def colorline(
        x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0),
        linewidth=2, alpha=0.8, linestyle='-'):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    if not hasattr(z, "__iter__"):
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)

    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                  linewidth=linewidth, alpha=alpha, linestyle=linestyle)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc

def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
