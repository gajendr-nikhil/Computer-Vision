"""Problem Set 7: Particle Filter Tracking."""

import numpy as np
import cv2
from collections import deque

import os

# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
class ParticleFilter(object):
    """A particle filter tracker, encapsulating state, initialization and update methods. Refer to the method
    run_particle_filter( ) in experiment.py to understand how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles. This should be a N x 2 array where
                                        N = self.num_particles. This component is used by the autograder so make sure
                                        you define it appropriately.
        - self.weights (numpy.array): Array of N weights, one for each particle.
                                      Hint: initialize them with a uniform normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video frame that will be used as the template to
                                       track.
        - self.frame (numpy.array): Current video frame from cv2.VideoCapture().

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame, values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track, values in [0, 255].
            kwargs: keyword arguments needed by particle filter model, including:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity measure.
                    - sigma_dyn (float): sigma value that can be used when adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y, width, and height values.
        """

        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame

        # Todo: Initialize your particles array. Read the docstring.
        self.nxpart = 1
        self.particles = np.zeros((self.nxpart * self.num_particles, 2))
        # Todo: Initialize your weights array. Read the docstring.
        self.weights = np.ones(self.nxpart * self.num_particles, dtype=np.float64) / (self.nxpart * self.num_particles)

        # Initialize any other components you may need when designing your filter.
        self.frame_number = 0
        self.ty, self.tx = self.template.shape[: 2]
        self.fy, self.fx = self.frame.shape[: 2]
        self.hy, self.wx = int(round((self.ty -1)/2)), int(round((self.tx -1)/2))
        self.scenter = (self.template_rect['y'] + self.hy, self.template_rect['x'] + self.wx)
        self.stWin = 150
        self.xf = np.array([np.random.choice(np.arange(self.scenter[0]-self.stWin,  self.scenter[0]+self.stWin), self.nxpart * self.num_particles, replace=True),
                            np.random.choice(np.arange(self.scenter[1]-self.stWin,  self.scenter[1]+self.stWin), self.nxpart * self.num_particles, replace=True)])

        self.gray_t = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.

        """

        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """

        return self.weights

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None (do not include a return call). This function
        should update the particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the image. This means you should address
        particles that are close to the image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        if self.frame_number <= 9:
            nump = self.nxpart * self.num_particles
        else:
            nump = self.num_particles
        xf_new = np.zeros_like(self.xf)
        w_new = np.copy(self.weights)
        for i in range(nump):
            sample = self.xf[:, np.random.choice(np.arange(nump), 1, replace=True, p=self.weights)[0]]
            while True:
                rnum = int(np.random.randn() * self.sigma_dyn + self.sigma_dyn/2.0)
                if (sample[1] + rnum) < int(self.template_rect['w'] // 2):
                    xf_new[:, i][1] = int(self.template_rect['w'] // 2)
                elif (sample[1] + rnum) >= int(self.fx - self.template_rect['w'] // 2):
                    xf_new[:, i][1] = int(self.fx - self.template_rect['w'] // 2)
                else:
                    xf_new[:, i][1] = (sample[1] + rnum)
                rnum = int(np.random.randn() * self.sigma_dyn + self.sigma_dyn/2.0)
                if (sample[0] + rnum) < int(self.template_rect['h'] // 2):
                    xf_new[:, i][0] = int(self.template_rect['h'] // 2)
                elif (sample[0] + rnum) >= int(self.fy - self.template_rect['h'] // 2):
                    xf_new[:, i][0] = int(self.fy - self.template_rect['h'] // 2)
                else:
                    xf_new[:, i][0] = (sample[0] + rnum)
                break
            self.particles[i] = xf_new[:, i]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            patch = gray[xf_new[:, i][0] - self.hy -1 : xf_new[:, i][0] + self.hy + 1, xf_new[:, i][1] - self.wx -1 : xf_new[:, i][1] + self.wx + 1]

            mse = np.mean(np.square(self.gray_t - patch), dtype=np.float64)

            w_new[i] = np.exp(-mse / (2 * self.sigma_exp))

        self.xf = xf_new
        self.weights = np.copy(w_new)
        if np.sum(self.weights) > 0.0:
            self.weights = self.weights / np.sum(self.weights)
        else:
            self.weights = np.ones(nump, dtype=np.float64) / (nump)

        self.frame_number += 1
        #print "frame number... %s" %self.frame_number

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model updates here!
        These steps will calculate the weighted mean. The resulting values should represent the
        tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay each successive
        frame with the following elements:

        - Every particle's (u, v) location in the distribution should be plotted by drawing a
          colored dot point on the image. Remember that this should be the center of the window,
          not the corner.
        - Draw the rectangle of the tracking window associated with the Bayesian estimate for
          the current location which is simply the weighted mean of the (u, v) of the particles.
        - Finally we need to get some sense of the standard deviation or spread of the distribution.
          First, find the distance of every particle to the weighted mean. Next, take the weighted
          sum of these distances and plot a circle centered at the weighted mean with this radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the particle filter.
        """
        u_weighted_mean = 0
        v_weighted_mean = 0

        for i in range(self.num_particles):
            u_weighted_mean += self.particles[i, 0] * self.weights[i]
            v_weighted_mean += self.particles[i, 1] * self.weights[i]
            cv2.circle(frame_in, (int(self.particles[i, 1]), int(self.particles[i, 0])), 2, (0,255,0), -1)
        top_left = int(v_weighted_mean - self.template_rect['w'] // 2), int(u_weighted_mean - self.template_rect['h'] // 2)
        bottom_right = int(v_weighted_mean + self.template_rect['w'] // 2), int(u_weighted_mean + self.template_rect['h'] // 2)
        cv2.rectangle(frame_in, top_left, bottom_right, (255,0,0), 2)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter object (parameters are the same as ParticleFilter).

        The documentation for this class is the same as the ParticleFilter above. There is one element that is added
        called alpha which is explained in the problem set documentation. By calling super(...) all the elements used
        in ParticleFilter will be inherited so you do not have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your implementation, you may comment out this
        function and use helper methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        if self.frame_number <= 9:
            nump = self.nxpart * self.num_particles
        else:
            nump = self.num_particles
        xf_new = np.zeros_like(self.xf)
        w_new = np.copy(self.weights)
        for i in range(nump):
            sample = self.xf[:, np.random.choice(np.arange(nump), 1, replace=True, p=self.weights)[0]]
            while True:
                rnum = int(np.random.randn() * self.sigma_dyn + self.sigma_dyn/2.0)
                if (sample[1] + rnum) < int(self.template_rect['w'] // 2):
                    xf_new[:, i][1] = int(self.template_rect['w'] // 2)
                elif (sample[1] + rnum) >= int(self.fx - self.template_rect['w'] // 2):
                    xf_new[:, i][1] = int(self.fx - self.template_rect['w'] // 2)
                else:
                    xf_new[:, i][1] = (sample[1] + rnum)
                rnum = int(np.random.randn() * self.sigma_dyn + self.sigma_dyn/2.0)
                if (sample[0] + rnum) < int(self.template_rect['h'] // 2):
                    xf_new[:, i][0] = int(self.template_rect['h'] // 2)
                elif (sample[0] + rnum) >= int(self.fy - self.template_rect['h'] // 2):
                    xf_new[:, i][0] = int(self.fy - self.template_rect['h'] // 2)
                else:
                    xf_new[:, i][0] = (sample[0] + rnum)
                break

            self.particles[i] = xf_new[:, i]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            patch = gray[xf_new[:, i][0] - self.hy -1 : xf_new[:, i][0] + self.hy + 1, xf_new[:, i][1] - self.wx -1 : xf_new[:, i][1] + self.wx + 1]

            mse = np.mean(np.square(self.gray_t - patch), dtype=np.float64)

            w_new[i] = np.exp(-mse / (2 * self.sigma_exp))

        self.xf = np.copy(xf_new)
        self.weights = np.copy(w_new)

        if np.sum(self.weights) > 0.0:
            self.weights = self.weights / np.sum(self.weights)
        else:
            self.weights = np.ones(nump, dtype=np.float64) / (nump)

        idx = self.weights.argmax()
        u_weighted_mean = self.particles[idx, 0]
        v_weighted_mean = self.particles[idx, 1]

        best_t = gray[int(u_weighted_mean) - self.hy - 1: int(u_weighted_mean) + self.hy + 1,
                      int(v_weighted_mean) - self.wx - 1: int(v_weighted_mean) + self.wx + 1]
        self.gray_t = self.alpha * best_t + (1 - self.alpha) * self.gray_t
        self.frame_number += 1
        #print "frame number... %s" %self.frame_number


class MDParticleFilter(ParticleFilter):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object (parameters same as ParticleFilter).

        The documentation for this class is the same as the ParticleFilter above.
        By calling super(...) all the elements used in ParticleFilter will be inherited so you
        don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.num_bins = kwargs.get('hist_bins_num', 8)  # required by the autograder
        self.vx = deque(20*[0], 20)
        self.uy = deque(20*[0], 20)
        self.prev_vx = 0
        self.prev_uy = 0
        self.xfs = np.array([np.random.choice(np.arange(self.scenter[0]-self.stWin,  self.scenter[0]+self.stWin), self.nxpart * self.num_particles, replace=True),
                             np.random.choice(np.arange(self.scenter[1]-self.stWin,  self.scenter[1]+self.stWin), self.nxpart * self.num_particles, replace=True),
                             np.random.choice(np.linspace(0.95, 0.993, num=800), self.nxpart * self.num_particles, replace=True)])
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def mslcpq(self, patch, template):
        hb_t = cv2.calcHist([template],[0],None,[self.num_bins],[0,256]).T
        hg_t = cv2.calcHist([template],[1],None,[self.num_bins],[0,256]).T
        hr_t = cv2.calcHist([template],[2],None,[self.num_bins],[0,256]).T
        h_t = np.hstack((hb_t, hg_t, hr_t))
        hb_p = cv2.calcHist([patch],[0],None,[self.num_bins],[0,256]).T
        hg_p = cv2.calcHist([patch],[1],None,[self.num_bins],[0,256]).T
        hr_p = cv2.calcHist([patch],[2],None,[self.num_bins],[0,256]).T
        h_p = np.hstack((hb_p, hg_p, hr_p))
        nr = np.square(h_t - h_p)
        dr = h_t + h_p
        dr[np.where(dr == 0.0)] = float('inf')
        return 0.5 * np.sum(nr / dr)

    def calc_mse(self, patch, template):
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        return np.mean(np.square(template - patch))

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your implementation, you may comment out this
        function and use helper methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """

        if self.frame_number <= 9:
            nump = self.nxpart * self.num_particles
        else:
            nump = self.num_particles
        xf_new = np.zeros_like(self.xfs)
        w_new = np.copy(self.weights)
        for i in range(nump):
            sample = self.xfs[:, np.random.choice(np.arange(nump), 1, replace=True, p=self.weights)[0]]
            while True:
                xf_new[:, i][2] = sample[2]

                rnum = int(np.random.randn() * self.sigma_dyn + self.sigma_dyn/2.0)
                if (sample[1] + rnum) < self.wx:
                    xf_new[:, i][1] = self.wx
                elif (sample[1] + rnum) >= int(self.fx - self.wx):
                    xf_new[:, i][1] = int(self.fx - self.wx)
                else:
                    xf_new[:, i][1] = (sample[1] + rnum)
                rnum = int(np.random.randn() * self.sigma_dyn + self.sigma_dyn/2.0)
                if (sample[0] + rnum) < self.hy:
                    xf_new[:, i][0] = self.hy
                elif (sample[0] + rnum) >= int(self.fy - self.hy):
                    xf_new[:, i][0] = int(self.fy - self.hy)
                else:
                    xf_new[:, i][0] = (sample[0] + rnum)
                break

            hy = int(xf_new[:, i][2] * self.hy) if int(xf_new[:, i][2] * self.hy) % 2 == 0 else int(xf_new[:, i][2] * self.hy) + 1
            wx = int(xf_new[:, i][2] * self.wx) if int(xf_new[:, i][2] * self.wx) % 2 == 0 else int(xf_new[:, i][2] * self.wx) + 1
            #hy = int(xf_new[:, i][2] * self.hy) if int(xf_new[:, i][2] * self.hy) % 2 != 0 else int(xf_new[:, i][2] * self.hy) + 1
            #wx = int(xf_new[:, i][2] * self.wx) if int(xf_new[:, i][2] * self.wx) % 2 != 0 else int(xf_new[:, i][2] * self.wx) + 1
            patch = frame[int(xf_new[:, i][0] - hy -1) : int(xf_new[:, i][0] + hy + 1),
                          int(xf_new[:, i][1] - wx -1) : int(xf_new[:, i][1] + wx + 1)].astype(np.float32)
            template = cv2.resize(self.template,(2*wx + 2, 2*hy + 2), interpolation = cv2.INTER_CUBIC).astype(np.float32)
            chi = self.mslcpq(patch, template)

            w_new[i] = np.exp(-chi / (2 * self.sigma_exp))

        if np.sum(w_new) > 1e-20:
            self.weights = w_new / np.sum(w_new)

            self.xfs = np.copy(xf_new)
            self.particles = np.copy(xf_new[:2, :].T)
            idx = self.weights.argmax()
            u_max, v_max = self.particles[idx, 0], self.particles[idx, 1]
            self.uy.appendleft(u_max - self.prev_uy)
            self.vx.appendleft(v_max - self.prev_vx)
            self.prev_uy, self.prev_vx = u_max, v_max
            self.hy = int(xf_new[:, i][2] * self.hy) if int(xf_new[:, i][2] * self.hy) % 2 == 0 else int(xf_new[:, i][2] * self.hy) + 1
            self.wx = int(xf_new[:, i][2] * self.wx) if int(xf_new[:, i][2] * self.wx) % 2 == 0 else int(xf_new[:, i][2] * self.wx) + 1
        else:
            self.xfs[0, :] = self.xfs[0, :] + (-0.2)#np.mean(np.array(self.uy)) / 2
            self.xfs[1, :] = self.xfs[1, :] #+ (-0.5)#np.mean(np.array(self.vx)) / 2
            self.particles = self.xfs[:2, :].T

        self.frame_number += 1
        #print "frame number... %s" %self.frame_number

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model updates here!
        These steps will calculate the weighted mean. The resulting values should represent the
        tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay each successive
        frame with the following elements:

        - Every particle's (u, v) location in the distribution should be plotted by drawing a
          colored dot point on the image. Remember that this should be the center of the window,
          not the corner.
        - Draw the rectangle of the tracking window associated with the Bayesian estimate for
          the current location which is simply the weighted mean of the (u, v) of the particles.
        - Finally we need to get some sense of the standard deviation or spread of the distribution.
          First, find the distance of every particle to the weighted mean. Next, take the weighted
          sum of these distances and plot a circle centered at the weighted mean with this radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the particle filter.
        """
        u_weighted_mean = 0
        v_weighted_mean = 0

        for i in range(self.num_particles):
            u_weighted_mean += self.particles[i, 0] * self.weights[i]
            v_weighted_mean += self.particles[i, 1] * self.weights[i]
            cv2.circle(frame_in, (int(self.particles[i, 1]), int(self.particles[i, 0])), 2, (0,255,0), -1)
        top_left = int(v_weighted_mean - self.wx), int(u_weighted_mean - self.hy)
        bottom_right = int(v_weighted_mean + self.wx), int(u_weighted_mean + self.hy)
        cv2.rectangle(frame_in, top_left, bottom_right, (255,0,0), 2)