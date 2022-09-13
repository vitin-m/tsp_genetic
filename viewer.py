import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import AnchoredText
import networkx as nx
import numpy as np

class Viewer:
    def __init__(
        self,
        gbests: np.ndarray,
        mean_fits: np.ndarray,
        std_devs: np.ndarray,
        dimension: int,
        edges: np.ndarray,
        iter: int,
        positions: dict,
    ):
        graph = nx.Graph()
        graph.add_nodes_from([x for x in range(dimension)])
        
        self.std_devs = std_devs
        self.mean_fits = mean_fits
        self.gbests = gbests
        self.graph = graph
        self.edges = edges
        self.positions = (self.get_positions(dimension, 1)
                          if positions == None else positions)
        
        self.frames = iter
        self.interval = 5000 / iter  # if 6000 / iter > 250 else 250
        
        self.fig = plt.figure("", figsize=(8, 8))
        self.axgrid = self.fig.add_gridspec(2, 2)

        self.ax_graphics = self.fig.add_subplot(self.axgrid[:, 0:1])

        self.ax_graphics.set_xlim(0, self.frames)
        self.ax_graphics.set_ylim(0, 1.2 * max(self.mean_fits))
        (self.line_gbest,) = self.ax_graphics.plot(0, 0)
        (self.line_mean,) = self.ax_graphics.plot(0, 0)
        (self.line_std,) = self.ax_graphics.plot(0, 0)

        self.ann = AnchoredText("", prop=dict(size=10), frameon=True, loc=2)
        self.ax_graphics.add_artist(self.ann)

        self.ax_graph = self.fig.add_subplot(self.axgrid[:, 1:])

    def update(self, i: int):
        self.ax_graph.clear()
        self.graph.remove_edges_from(list(self.graph.edges))
        iter = list(range(i))
        self.ann.txt.set_text(f'Generation:{i}\n'
                              f'Current Best Fitness: {self.gbests[i]}\n'
                              f'Current Mean Fitness: {self.mean_fits[i]:.1f}')
        self.line_gbest.set_xdata(iter)
        self.line_gbest.set_ydata(list(self.gbests[:i]))

        self.line_mean.set_xdata(iter)
        self.line_mean.set_ydata(list(self.mean_fits[:i]))
        
        self.line_std.set_xdata(iter)
        self.line_std.set_ydata(list(self.std_devs[:i]))
        #edges = self.edges[randint(0,1)]

        edges = self.edges[i]
        for j in range(len(edges)):
            self.graph.add_edges_from([edges[j]])

        nx.draw_networkx(self.graph, ax=self.ax_graph, pos=self.positions)
        return self.line_gbest

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def show(self):
        self.animation = FuncAnimation(
            self.fig, func=self.update, frames=self.frames, interval=self.interval
        )
        self.paused = False
        self.fig.canvas.mpl_connect("button_press_event", self.toggle_pause)
        plt.show()

def get_edges(individual : np.ndarray):
    edges = [(individual[i - 1], individual[i]) for i in range(1, len(individual))]
    return edges

def get_positions(vertexCount : int, radius: float):
    positions = dict()
    div = math.radians(360 / vertexCount)
    rad = 0

    for i in range (vertexCount + 1):
        x = math.cos(rad) * radius
        y = math.sin(rad) * radius
        positions[i] = (x,y)
        rad += div

    return positions

def main():
    gbests = np.arange(0, 1000, 10)
    mean_fitness = np.arange(100)
    max_iter = 100

    vertex = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    G = nx.Graph()
    G.add_nodes_from(vertex)

    pos = get_positions(len(vertex), 1)
    edges = np.array([[(1,2), (3,4), (5,1), (3,2)], [(3,5),(5,4),(1,2),(2,5)]])
    vr = Viewer(gbests, mean_fitness, G, edges, pos, max_iter)
    vr.show()


if __name__ == "__main__":
    main()
