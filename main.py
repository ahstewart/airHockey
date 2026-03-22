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
    # math.atan2 natively handles all quadrants and x=0 edge cases
    return math.degrees(math.atan2(vector[1], vector[0])) % 360


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
                # 1. Snap the puck to the wall to prevent clipping
                puck_x = float(self.puck_diameter/2)
                norm_x = 1
                norm_y = 0
                #print(f"v = {v_x, v_y}")
                ref_x, ref_y = calc_reflect((v_x,v_y), (norm_x,norm_y))
                shot_angle = calc_angle((ref_x, ref_y))
                #print("hit left wall")
                #print(f"x = {puck_x}, ref_vec = {ref_x, ref_y}")
            elif puck_x >= (self.table_x - float(self.puck_diameter/2)):
                # 1. Snap the puck to the wall to prevent clipping
                puck_x = (self.table_x - float(self.puck_diameter/2))
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
                # Snap the puck to the wall to prevent clipping
                puck_x = float(self.puck_diameter/2)
                norm_x = 1
                norm_y = 0
                ref_x, ref_y = calc_reflect((v_x, v_y), (norm_x, norm_y))
                shot_angle = calc_angle((ref_x, ref_y))
                temp_v_x = math.cos(math.radians(shot_angle))
                temp_v_y = math.sin(math.radians(shot_angle))
                #print(f"hit left wall\nv={v_x, v_y}, ref={ref_x, ref_y}, angle={shot_angle},pos={puck_x, puck_y}")
            elif puck_x >= (self.table_x - float(self.puck_diameter/2)):
                # Snap the puck to the wall to prevent clipping
                puck_x = (self.table_x - float(self.puck_diameter/2))
                norm_x = -1
                norm_y = 0
                ref_x, ref_y = calc_reflect((v_x, v_y), (norm_x, norm_y))
                shot_angle = calc_angle((ref_x, ref_y))
                temp_v_x = math.cos(math.radians(shot_angle))
                temp_v_y = math.sin(math.radians(shot_angle))
                #print(f"hit right wall\nv={v_x, v_y}, ref={ref_x, ref_y}, angle={shot_angle}, pos={puck_x, puck_y}")
            # check if puck hits pusher
            elif ((puck_x - pusher_x)**2 + (puck_y - pusher_y)**2) <= ((self.puck_diameter + self.pusher_diameter) / 2)**2:
                dist = math.hypot((puck_x - pusher_x), (puck_y - pusher_y))
                min_dist = (self.puck_diameter + self.pusher_diameter) / 2
                
                # 1. Snap puck OUT of the pusher to prevent infinite clipping loops
                # We add a tiny 0.01 epsilon to ensure it completely clears the boundary
                overlap = min_dist - dist
                puck_x += ((puck_x - pusher_x) / dist) * (overlap + 0.01)
                puck_y += ((puck_y - pusher_y) / dist) * (overlap + 0.01)

                # 2. The normal vector is simply the line connecting the two centers
                norm_x, norm_y = norm_vector([puck_x - pusher_x, puck_y - pusher_y])

                # 3. Reflect the velocity
                ref_x, ref_y = calc_reflect((v_x, v_y), (norm_x, norm_y))
                shot_angle = calc_angle((ref_x, ref_y))
                #print("hit pusher")
            # check if puck hits back wall
            elif puck_y >= (self.table_y - float(self.puck_diameter / 2)):
                # Snap the puck to the wall to prevent clipping
                puck_y = (self.table_y - float(self.puck_diameter / 2))
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
        
    def shoot_vectorized(self, pusher_x, pusher_y, shots):
        # Initialize arrays for all pucks simultaneously
        puck_x = np.random.uniform(0, self.table_x, shots)
        puck_y = np.random.uniform(0, self.table_y / 2, shots)
        shot_angle = np.random.uniform(0.01, 180, shots)
        
        # Calculate initial velocities
        v_x = np.cos(np.radians(shot_angle))
        v_y = np.sin(np.radians(shot_angle))
        
        # State tracking arrays
        active = np.ones(shots, dtype=bool)       # True if puck is still in play
        results = np.zeros(shots, dtype=int)      # 0: timeout, 1: miss, 2: goal
        passed_median = np.zeros(shots, dtype=bool) 
        
        r_puck = float(self.puck_diameter / 2)
        r_pusher = float(self.pusher_diameter / 2)
        min_dist_sq = (r_puck + r_pusher) ** 2
        
        # Use a max loop count instead of time.time() to define a timeout limit.
        # ~5000 loops is roughly equivalent to a few seconds of simulated movement.
        max_loops = 5000 
        loops = 0
        
        while np.any(active) and loops < max_loops:
            loops += 1
            
            # 1. Update positions for ALL active pucks at once
            puck_x[active] += self.rate * v_x[active]
            puck_y[active] += self.rate * v_y[active]
            
            # 2. Check Left Wall
            hit_left = active & (puck_x <= r_puck)
            puck_x[hit_left] = r_puck
            v_x[hit_left] *= -1
            
            # 3. Check Right Wall
            hit_right = active & (puck_x >= self.table_x - r_puck)
            puck_x[hit_right] = self.table_x - r_puck
            v_x[hit_right] *= -1
            
            # 4. Update Median Status
            just_passed = active & ~passed_median & (puck_y >= self.table_y / 2)
            passed_median[just_passed] = True
            
            # 5. Check Pusher Collisions (Only for pucks past median)
            dist_sq = (puck_x - pusher_x)**2 + (puck_y - pusher_y)**2
            hit_pusher = active & passed_median & (dist_sq <= min_dist_sq)
            
            if np.any(hit_pusher):
                # Snap pucks out of the pusher
                dist = np.sqrt(dist_sq[hit_pusher])
                overlap = (r_puck + r_pusher) - dist + 0.01
                
                # Calculate normal vectors
                nx = (puck_x[hit_pusher] - pusher_x) / dist
                ny = (puck_y[hit_pusher] - pusher_y) / dist
                
                puck_x[hit_pusher] += nx * overlap
                puck_y[hit_pusher] += ny * overlap
                
                # Reflect velocities using vector dot product: v_new = v - 2(v.n)n
                dot = v_x[hit_pusher] * nx + v_y[hit_pusher] * ny
                v_x[hit_pusher] = v_x[hit_pusher] - 2 * dot * nx
                v_y[hit_pusher] = v_y[hit_pusher] - 2 * dot * ny
            
            # 6. Check Back Wall
            hit_back = active & passed_median & (puck_y >= self.table_y - r_puck)
            if np.any(hit_back):
                puck_y[hit_back] = self.table_y - r_puck
                v_y[hit_back] *= -1
            
            # 7. Goal Check (Hit back wall AND inside goal width)
            is_goal = hit_back & (puck_x > self.goal_x_1 + r_puck) & (puck_x < self.goal_x_2 - r_puck)
            results[is_goal] = 2
            active[is_goal] = False
            
            # 8. Miss Check (Bounced back past median)
            is_miss = active & passed_median & (puck_y <= self.table_y / 2)
            results[is_miss] = 1
            active[is_miss] = False

        # Calculate success rate, excluding timeouts (results == 0)
        total_valid = np.sum(results > 0)
        if total_valid == 0:
            return 0.0
        
        goals = np.sum(results == 2)
        return float(goals / total_valid)

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
            pusher_x = float(self.pusher_diameter/2)
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
                table[j, i] = rate
                pusher_x += step
                #signal.alarm(0)
            pusher_y -= step
        return table
    
    def monte_carlo_vec(self, shots=10000, step=1):
        table_width = self.table_x
        table_length = float(self.table_y / 2)
        
        pusher_x = float(self.pusher_diameter / 2)
        pusher_y = self.table_y - float(self.pusher_diameter / 2)
        
        array_table_width = int((table_width - self.pusher_diameter) / step) + 1
        array_table_length = int((table_length - self.pusher_diameter) / step) + 1
        
        table = np.zeros((array_table_width, array_table_length))
        
        print("Running Vectorized Monte Carlo...")
        for i in tqdm(range(array_table_length)):
            pusher_x = float(self.pusher_diameter / 2) # Reset X for the new row
            
            for j in range(array_table_width):
                # Let numpy handle all shots for this coordinate simultaneously
                rate = self.shoot_vectorized(pusher_x, pusher_y, shots)
                
                # Store in table (Note: j is width/x-axis, i is length/y-axis)
                table[j, i] = rate
                
                pusher_x += step
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

    def plot_heatmap(self, table):
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
        # find padding size
        pad_width = int((self.table_x - table.shape[0])/2)
        pad_height = int((self.table_y/2 - table.shape[1])/2)
        table = table.T
        table_padded = np.pad(table, pad_width=((pad_width, pad_height), (pad_height, pad_width)), mode='constant', constant_values=0)
        # draw heatmap
        heatmap = self.ax.imshow(table_padded, cmap='viridis', interpolation='nearest')
        self.fig.colorbar(heatmap, ax=self.ax)
        plt.show()




if __name__ == "__main__":
    print("Making Puck() object called sim... use .monte_carlo method to run simulation")
    sim = Puck()
    results = sim.monte_carlo_vec(shots=1000, step=1)
    sim.plot_heatmap(results)