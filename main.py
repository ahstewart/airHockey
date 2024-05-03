import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import collections
import math
from tqdm import tqdm
import time
import signal


def calc_reflect(vel_vec, norm_vec):
    # calculates the resulting velocity vector after being reflected across the normal vector
    dot = np.dot(vel_vec, norm_vec)
    return (vel_vec[0]-(2*dot*norm_vec[0])), (vel_vec[1]-(2*dot*norm_vec[1]))


def check_quadrant(coords):
    # returns the quadrant the given coordinate is located in
    # quadrant 1: x and y are positive
    if coords[0] >= 0 and coords[1] >= 0:
        return 1
    elif coords[0] <= 0 and coords[1] >= 0:
        return 2
    elif coords[0] <= 0 and coords[1] <= 0:
        return 3
    elif coords[0] >= 0 and coords[1] <= 0:
        return 4


def calc_angle(vector):
    # calculates an angle given a vector starting at the origin
    if vector[1] == 0:
        angle = 90
    else:
        angle = math.degrees(math.atan(abs(float(vector[1]/vector[0]))))
    quad = check_quadrant(vector)
    # print(f"angle = {angle}, quadrant = {quad}")
    if quad == 2:
        angle = 180 - angle
    elif quad == 3:
        angle += 180
    elif quad == 4:
        angle = 360 - angle
    return angle


def norm_vector(vec):
    mag = np.linalg.norm(vec)
    return float(vec[0]/mag), float(vec[1]/mag)


class Puck:
    def __init__(self):
        # table dimensions: x=width and y=length in centimeters
        self.table_x = 127
        self.table_y = 243
        self.puck_diameter = 7.5
        self.pusher_diameter = 9.6
        self.goal_width = float(self.table_x/4)
        self.goal_x_1 = float(self.table_x/2) - float(self.goal_width/2)
        self.goal_x_2 = float(self.table_x/2) + float(self.goal_width/2)
        self.pusher_x_init = 0
        self.pusher_y_init = 0
        self.puck_x_init = 0
        self.puck_y_init = 0
        # list used to store all positional values of the puck
        self.puck_pos = []
        # rate of puck progression, in units used above
        self.rate = 0.5
        # build initial plots
        self.fig, self.ax = plt.subplots()
        self.puck_patches = []
        # set timeout value in seconds
        self.timeout = 5

    def shoot(self, pusher_loc=(0, 0)):
        # shot_result will be returned as 0, 1, or 2. 0 means the shot timeout was exceeded, so there was some failure
        # in the simulation. 1 means the shot was a miss. 2 means the shot was a goal.
        # start with a fresh shot
        self.puck_pos = []
        # define the start of the shot - the angle and the starting location
        shot_angle = np.random.uniform(0.01, 180)
        v_x = math.cos(math.radians(shot_angle))
        v_y = math.sin(math.radians(shot_angle))
        puck_x = np.random.uniform(0, self.table_x)
        self.puck_x_init = puck_x
        puck_y = np.random.uniform(0, self.table_y/2)
        self.puck_y_init = puck_y
        self.puck_pos.append((puck_x, puck_y))
        #print(puck_x, puck_y)
        #print(f"initial angle {shot_angle}")
        # define location of pusher
        if pusher_loc == (0, 0):
            pusher_x = np.random.uniform(self.puck_diameter, (self.table_x - self.puck_diameter))
            self.pusher_x_init = pusher_x
            pusher_y = np.random.uniform(self.table_y/2, self.table_y - self.puck_diameter)
            self.pusher_y_init = pusher_y
        else:
            if (float(self.pusher_diameter/2) <= pusher_loc[0] <= (self.table_x - float(self.pusher_diameter/2))) \
                    and ((float(self.table_y/2) + float(self.pusher_diameter/2)) <= pusher_loc[1] <=
                         (self.table_y - float(self.pusher_diameter/2))):
                pusher_x = pusher_loc[0]
                self.pusher_x_init = pusher_x
                pusher_y = pusher_loc[1]
                self.pusher_y_init = pusher_y
            else:
                print("Error: Using a pusher location outside of the table.")
                return None
        # move puck at defined rate until it crosses median
        start_time = time.time()
        while puck_y < float(self.table_y/2):
            v_x = math.cos(math.radians(shot_angle))
            v_y = math.sin(math.radians(shot_angle))
            puck_x += self.rate * v_x
            puck_y += self.rate * v_y
            # check if puck hits wall - if it does, mirror the shot angle
            if puck_x <= (float(self.puck_diameter/2)):
                norm_x = 1
                norm_y = 0
                #print(f"v = {v_x, v_y}")
                ref_x, ref_y = calc_reflect((v_x,v_y), (norm_x,norm_y))
                shot_angle = calc_angle((ref_x, ref_y))
                #print("hit left wall")
                #print(f"x = {puck_x}, ref_vec = {ref_x, ref_y}")
            elif puck_x >= (self.table_x - float(self.puck_diameter/2)):
                norm_x = -1
                norm_y = 0
                #print(f"v = {v_x,v_y}")
                ref_x, ref_y = calc_reflect((v_x, v_y), (norm_x, norm_y))
                shot_angle = calc_angle((ref_x, ref_y))
                #print("hit right wall")
                #print(f"x = {puck_x}, ref_vec = {ref_x, ref_y}")
            #print("hasn't passed")
            self.puck_pos.append((puck_x, puck_y))
            #print(puck_x, puck_y)
            # check if timeout has been reached
            run_time = time.time() - start_time
            if run_time > self.timeout:
                print("Shot failed: Timeout exceeded. Didn't cross median")
                return 0
        # once it's over the median, move puck at defined rate until it hits the other end or bounces back
        puck_y_prev = puck_y
        #print("passed median")
        start_time = time.time()
        while not (((puck_y >= (self.table_y-float(self.puck_diameter/2))) and
               ((self.goal_x_1+float(self.puck_diameter/2)) < puck_x < (self.goal_x_2-float(self.puck_diameter/2))))
                or (puck_y <= float(self.table_y/2))):
            #print("hasn't scored or passed median")
            puck_y_prev = puck_y
            v_x = math.cos(math.radians(shot_angle))
            v_y = math.sin(math.radians(shot_angle))
            puck_x += self.rate * math.cos(math.radians(shot_angle))
            puck_y += self.rate * math.sin(math.radians(shot_angle))
            # check if puck hits left or right wall - if it does, mirror the shot angle
            if puck_x <= (float(self.puck_diameter/2)):
                norm_x = 1
                norm_y = 0
                ref_x, ref_y = calc_reflect((v_x, v_y), (norm_x, norm_y))
                shot_angle = calc_angle((ref_x, ref_y))
                temp_v_x = math.cos(math.radians(shot_angle))
                temp_v_y = math.sin(math.radians(shot_angle))
                while puck_x <= (float(self.puck_diameter / 2)):
                    #print("getting puck back inside")
                    puck_x += self.rate * temp_v_x
                    puck_y += self.rate * temp_v_y
                    #print(temp_v_x, temp_v_y)
                    #print(puck_x, puck_y)
                    if run_time > self.timeout:
                        print("Shot failed: Timeout exceeded. Passed median.")
                        return 0
                #print(f"hit left wall\nv={v_x, v_y}, ref={ref_x, ref_y}, angle={shot_angle},pos={puck_x, puck_y}")
            elif puck_x >= (self.table_x - float(self.puck_diameter/2)):
                norm_x = -1
                norm_y = 0
                ref_x, ref_y = calc_reflect((v_x, v_y), (norm_x, norm_y))
                shot_angle = calc_angle((ref_x, ref_y))
                temp_v_x = math.cos(math.radians(shot_angle))
                temp_v_y = math.sin(math.radians(shot_angle))
                while puck_x >= (self.table_x - float(self.puck_diameter/2)):
                    #print("getting puck back inside")
                    puck_x += self.rate * temp_v_x
                    puck_y += self.rate * temp_v_y
                    #print(temp_v_x, temp_v_y)
                    #print(puck_x, puck_y)
                    run_time = time.time() - start_time
                    if run_time > self.timeout:
                        print("Shot failed: Timeout exceeded. Passed median.")
                        return 0
                #print(f"hit right wall\nv={v_x, v_y}, ref={ref_x, ref_y}, angle={shot_angle}, pos={puck_x, puck_y}")
            # check if puck hits pusher
            elif (float(self.puck_diameter/2)-float(self.pusher_diameter/2))**2 <= ((puck_x - pusher_x)**2 + (puck_y - pusher_y)**2) <= ((float(self.puck_diameter/2))+float(self.pusher_diameter/2))**2:
                d = math.hypot((puck_x-pusher_x), (puck_y-pusher_y))
                l = (float(self.puck_diameter/2)**2 - float(self.pusher_diameter/2)**2 + d**2) / (2*d)
                h = np.sqrt(float(self.puck_diameter/2)**2 - l**2)
                x_intersect_1 = (l / d) * (pusher_x-puck_x) + (h / d) * (pusher_y-puck_y) + puck_x
                x_intersect_2 = (l / d) * (pusher_x-puck_x) - (h / d) * (pusher_y-puck_y) + puck_x
                y_intersect_1 = (l / d) * (pusher_y-puck_y) - (h / d) * (pusher_x-puck_x) + puck_y
                y_intersect_2 = (l / d) * (pusher_y-puck_y) + (h / d) * (pusher_x-puck_x) + puck_y
                y_intersect_diff = y_intersect_2 - y_intersect_1
                x_intersect_diff = x_intersect_2 - x_intersect_1
                # detect whether puck is above pusher or below
                position = ((x_intersect_2 - x_intersect_2) * (puck_y - y_intersect_1) - (y_intersect_2 - y_intersect_1) * (puck_x - x_intersect_1))
                # if it does hit, reflect shot angle off of pusher
                norm_x = 1
                norm_y = float(-1 * x_intersect_diff / y_intersect_diff)
                norm_x, norm_y = norm_vector([norm_x, norm_y])
                if position > 0:
                    if norm_y < 0:
                        norm_x *= -1
                        norm_y *= -1
                else:
                    if norm_y > 0:
                        norm_x *= -1
                        norm_y *= -1
                ref_x, ref_y = calc_reflect((v_x, v_y), (norm_x, norm_y))
                shot_angle = calc_angle((ref_x, ref_y))
                #print("hit pusher")
            # check if puck hits back wall
            elif puck_y >= (self.table_y - float(self.puck_diameter / 2)):
                norm_x = 0
                norm_y = -1
                ref_x, ref_y = calc_reflect((v_x, v_y), (norm_x, norm_y))
                shot_angle = calc_angle((ref_x, ref_y))
                temp_v_x = math.cos(math.radians(shot_angle))
                temp_v_y = math.sin(math.radians(shot_angle))
                #puck_x += self.rate * temp_v_x
                #puck_y += self.rate * temp_v_y
                #print(f"hit back wall\npos={puck_x, puck_y}, ref={ref_x, ref_y}, angle={shot_angle}")
            self.puck_pos.append((puck_x, puck_y))
            run_time = time.time() - start_time
            if run_time > self.timeout:
                print("Shot failed: Timeout exceeded. Passed median.")
                return 0
            #print(puck_x, puck_y)
        #print(puck_x, puck_y)
        if (self.table_y - (1.5*self.puck_diameter)) < puck_y < (self.table_y + (1.5*self.puck_diameter)):
            #print("Goal!")
            return 2
        elif (float(self.table_y/2) - (1.5*self.puck_diameter)) < puck_y < (float(self.table_y/2) + (1.5*self.puck_diameter)):
            #print("Miss!")
            return 1

    def monte_carlo(self, shots=10000, step=1):
        # runs a monte carlo simulation where for a given pusher position (x, y), a number of shots are attempted
        # after that many pucks are shot, the make/total ratio is calculated and the pusher is moved to another position
        # determined by the step parameter
        # set timeout handler
        #signal.signal(signal.SIGINT, self.handler)
        table_width = self.table_x
        table_length = float(self.table_y/2)
        pusher_x = float(self.pusher_diameter/2)
        pusher_y = self.table_y - float(self.pusher_diameter/2)
        # define array of upper half of table
        array_table_width = 0
        array_table_length = 0
        x = 0
        y = 0
        while x <= (table_width - self.pusher_diameter):
            x += step
            array_table_width += 1
        while y <= (table_length - self.pusher_diameter):
            y += step
            array_table_length += 1
        table = np.zeros((array_table_width, array_table_length))
        for i in tqdm(range(array_table_length-1)):
            for j in (range(array_table_width-1)):
                results = []
                for s in range(shots):
                    #signal.alarm(self.timeout)
                    try:
                        r = self.shoot(pusher_loc=(pusher_x, pusher_y))
                        if r == 1:
                            results.append(0)
                        elif r == 2:
                            results.append(1)
                    except Exception as exc:
                        print(exc)
                goals = np.sum(results)
                rate = float(goals/len(results))
                #print(rate)
                table[i, j] = rate
                pusher_x += step
                #signal.alarm(0)
            pusher_y -= step
        return table

    # def handler(self, signum, frame):
    #     print("Error: Puck shot timed out.")
    #     raise Exception("something went wonky")

    def animate_func(self, frame):
        # draw puck
        self.ax.add_patch(self.puck_patches[frame])
        # if this isn't the first frame, remove previous circle
        if 0 < frame < len(self.puck_pos)-1:
            self.puck_patches[frame-1].remove()

    def animate(self, suffix):
        # clear previous plot
        plt.clf()
        # add new plot
        self.fig, self.ax = plt.subplots()
        # store table boundaries in line list
        lines = []
        # draw left side of table - left will be zero-valued x-axis
        lines.append([(0, 0), (0, self.table_y)])
        # draw right side of table
        lines.append([(self.table_x, 0), (self.table_x, self.table_y)])
        # draw median line
        lines.append([(0, float(self.table_y / 2)), (self.table_x, float(self.table_y / 2))])
        # draw upper edge of table
        lines.append([(0, self.table_y), (self.table_x, self.table_y)])
        # draw bottom edge of table
        lines.append([(0, 0), (self.table_x, 0)])
        lc = collections.LineCollection(lines, color='black')
        self.ax.add_collection(lc)
        # set limits on axes
        self.ax.set(xlim=[-50, self.table_y + 50], ylim=[-50, self.table_y + 50])
        # draw goals
        goals = [[(self.goal_x_1, 0), (self.goal_x_2, 0)], [(self.goal_x_1, self.table_y), (self.goal_x_2, self.table_y)]]
        gc = collections.LineCollection(goals, color='blue')
        self.ax.add_collection(gc)
        # draw pusher location
        draw_pusher = plt.Circle((self.pusher_x_init, self.pusher_y_init), float(self.pusher_diameter / 2),
                                 color='yellow')
        self.ax.add_patch(draw_pusher)
        # get list of puck patches from puck positions
        for f in range(len(self.puck_pos)):
            self.puck_patches.append(plt.Circle(self.puck_pos[f], float(self.puck_diameter / 2), color='red'))
        # animate the shot
        print("Animating the shot...")
        ani = animation.FuncAnimation(fig=self.fig, func=self.animate_func, frames=tqdm(range(len(self.puck_pos))),
                                      interval=5, blit=False)
        ani.save(f"ani_{suffix}.mp4")
        plt.show()


if __name__ == "__main__":
    print("Making Puck() object called sim... use .monte_carlo method to run simulation")
    sim = Puck()