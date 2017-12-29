import theano
import theano.tensor as T

import numpy as np

import time
import matplotlib.pyplot as plt

plt.ion()

# cool seeds:
# seed, description - network model
# 919333390 - 1
# 650867666 - 1

# 74214537, overlapping orange blobs - 1

# 112940321 - 2

# 525058265 - 3

seed = raw_input("Choose a seed, press enter for a random one: ")

if seed == "":
    seed = np.random.randint(0, 1000000000)

print("This is the network seed: " + str(seed))
np.random.seed(int(seed))

weight1 = theano.shared(0.2 * np.random.randn(2, 10) * 1j + 0.2 * np.random.randn(2, 10))
weight2 = theano.shared(0.35 * np.random.randn(10, 10) * 1j + 0.35 * np.random.randn(10, 10))
weight3 = theano.shared(0.45 * np.random.randn(10, 1) * 1j + 0.45 * np.random.randn(10, 1))

def model(inp):
    """
        MODEL 1:
        fc1 = T.dot(inp, weight1) ** np.sqrt(2)
        fc2 = T.tan(T.dot(fc1, weight2))
        fc3 = T.dot(fc2, weight3)
    """

    """
        MODEL 2:
        This model is particularly interesting because of the piecewise nature of the non-linearity (max). This results in fragments or shards in the end fractal.

        fc1 = T.maximum(T.dot(inp, weight1) ** np.sqrt(2), 0)
        fc2 = T.maximum(T.tan(T.dot(fc1, weight2)), 0)
        fc3 = T.dot(fc2, weight3)
    """

    """
        MODEL 3:
    """

    fc1 = T.dot(T.minimum(inp, 0), weight1) ** np.sqrt(2)
    fc2 = T.tan(T.dot(fc1, weight2))
    fc3 = T.dot(fc2, weight3)

    return fc3

inp = T.zmatrix()
output = model(inp)

nn = theano.function(inputs = [inp], outputs = [output])

z = None
c = None
heatmap = None

def graph(x_min, x_max, y_min, y_max, step, graph = True):
    """
        graph(...) -> None

        Given window parameters, graphs the fractal with 30 iterations and a threshold of 0.8
    """

    global z, c, heatmap

    num_x = (x_max - x_min)/step
    num_y = (y_max - y_min)/step

    z = np.zeros(int(num_x) * int(num_y)) * (1j + 1)
    heatmap = np.zeros(int(num_x) * int(num_y))

    # Yikes
    s = time.time()
    c = []
    for x in range (0, int(num_x)):
        x_ = x_min + x * step
        temp = []
        for y in range (0, int(num_y)):
            y_ = y_min + y * step
            temp.append(x_ * (1j + 0) + y_ * (0j + 1))

        c.append(temp)

    c = np.array(c).flatten()
    print(time.time() - s)


    for iteration in range (30):
        # housecleaning for neural network input
        c_ = np.expand_dims(c, -1)
        z_ = np.expand_dims(z, -1)

        inp = np.concatenate([c_, z_], axis = -1)

        # remove the second axis
        z = nn(inp)[0][:, 0]

        # is z above the threshold?
        heatmap += np.absolute(z) > 0.8

    # plot it
    plt.cla()
    plt.imshow(heatmap.reshape([int(num_x), int(num_y)]), cmap = "jet")
    plt.pause(0.001)

def iteration(x_min, x_max, y_min, y_max, step, graph = True, its = 1):
    """
        iteration(...) -> None

        Performs a single iteration of the recursive function and updates the heatmap
    """

    global z, c, heatmap

    num_x = (x_max - x_min)/step
    num_y = (y_max - y_min)/step

    for i in range (its):
        print(i)
        c_ = np.expand_dims(c, -1)
        z_ = np.expand_dims(z, -1)

        #print(c_.shape, z_.shape)

        inp = np.concatenate([c_, z_], axis = -1)

        z = nn(inp)[0][:, 0]

        heatmap += np.absolute(z) > 0.8

    plt.cla()
    plt.imshow(heatmap.reshape([int(num_x), int(num_y)]), cmap = "jet")
    plt.pause(0.001)

# initial window parameters, adjust as needed
x_min = -4
x_max = 4
y_min = -4
y_max = 4

step = 0.025

window = 4

num_x = (x_max - x_min)/step
num_y = (y_max - y_min)/step

fig, ax = plt.subplots()

def onclick(event):
    global x_min, x_max, y_min, y_max, step, window, num_x, num_y

    # zoom, first by centering around the mouse click
    posx = (event.ydata / num_x) * (x_max - x_min) + x_min
    posy = (event.xdata / num_y) * (y_max - y_min) + y_min

    # then decrease the window size (zoom)
    if (window == 2.):
        window = 0.5
        step = 0.01
    else:
        window /= 2.

    # center the window around posx and posy
    x_min = posx - window
    x_max = posx + window

    y_min = posy - window
    y_max = posy + window

    # increase resolution by factor of two (computation time is maintained because you also scaled the size of the window)
    step /= 2.

    # debugging stuff, not needed
    num_x = (x_max - x_min)/step
    num_y = (y_max - y_min)/step
    print(num_x, num_y)

    # re-graph
    graph(x_min, x_max, y_min, y_max, step)

# event listener for clicks
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# really bad command line for refining or adding extra iterations to the graph
while True:
    cmd = raw_input("> ")

    s = cmd.split(' ')
    if (s[0] == "iterate"):
        its = int(s[1])
        iteration(x_min, x_max, y_min, y_max, step, its = its)

    elif (s[0] == "refine"):
        factor = float(s[1])
        step /= factor
        graph(x_min, x_max, y_min, y_max, step)
