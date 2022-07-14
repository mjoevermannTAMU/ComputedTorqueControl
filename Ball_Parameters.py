import numpy as np

# this file will define the ball EOM as presented in the Kayacan paper
class Ball_Robot():
    # ball characteristics shared by all instances
    Ms = 5
    mp = 10
    r = 0.5
    R = 1
    Ip = (Ms+mp)*r**2
    Is = (mp*r**2)
    g = 9.81
    def __init__(self):# define variables that track the balls state unique to this instance
        self.ball_drive_angle = 0.0  # global drive angle
        self.pend_drive_angle = 0.0  # pendulum angle w.r.t. vertical measured with the vn
        self.ball_pipe_angle = np.pi / 2  # pipe angle w.r.t. vertical measured with the vn
        self.steer_angle = 0.0  # steering angle of the pendulum, measured from perpendicular to the pipe


        # store previous measurements to estimate velocity
        self.prev_pend_drive_angle = 0.0
        self.prev_ball_pipe_angle = np.pi / 2


    # define some updaters
    def update_ball_drive(self,x):
        self.ball_drive_angle = x
    def update_pend_drive(self, x):
        self.pend_drive_angle = x
    def update_pipe_angle(self, x):
        self.ball_pipe_angle = x
    def update_steer(self, x):
        self.steer_angle = x
    def update_prev_pend_drive(self,x):
        self.prev_pend_drive_angle = x
    def update_prev_pipe_angle(self, x):
        self.prev_ball_pipe_angle = x


    # define methods to output ball state mtx's
    def get_M(self, q): #[sphere driving, pendulum drive, pipe angle, steering angle]
        q1, q2, q3, q4 = q
        M11 = self.Ms*self.R**2 + self.mp*self.R**2 +self.mp*self.r**2 + self.Is + self.Ip + 2*self.mp*self.R*self.r*np.cos(q2-q1)
        M12, M21 = -self.mp*self.r**2 - self.Ip - self.mp*self.R*self.r*np.cos(q2-q1)
        M13, M14, M23, M24, M31, M32, M41, M42 = 0,0,0,0,0,0,0,0
        M22 = self.mp*self.r**2 + self.Ip
        M33 = self.Ms*self.R**2 + self.mp*self.R**2 + self.mp*self.r**2 + self.Is + self.Ip + 2*self.mp*self.R*self.r*np.cos(q3-q4)
        M34, M43 = -self.mp*self.r**2 - self.Ip - self.mp*self.R*self.r*np.cos(q4-q3)
        M44 = self.mp*self.r**2 + self.Ip

        return np.array([[M11, M12,   0,   0],
                         [M21, M22,   0,   0],
                         [  0,   0, M33, M34],
                         [  0,   0, M43, M44]])

    def get_V(self, q, q_dot):  # [sphere driving, pendulum drive, pipe angle, steering angle]

        q1, q2, q3, q4 = q
        q1d, q2d, q3d, q4d = q_dot
        V11 = self.mp*self.R*self.r*np.sin(q2-q1)*(q1d**2 + q2d**2 - 2*q1*q2) - self.mp*self.g*self.r*np.sin(q2-q1)
        V21 = self.mp*self.g*self.r*np.sin(q2-q1)
        V31 = self.mp*self.R*self.r*np.sin(q4-q3)*(q4d**2 + q3d**2 - 2*q4*q3) - self.mp*self.g*self.r*np.sin(q4-q3)
        V41 = self.mp * self.g * self.r * np.sin(q4 - q3)
        return np.array([[V11],
                        [V21],
                        [V31],
                        [V41]])
    def get_M_static(self,q):
        return np.array([[self.Ip, 0],
                        [0, self.Is]])
    def get_V_static(self, q):
        return self.mp*self.g*self.r*np.array([[np.sin(q[0])],
                                               [np.sin(q[1])]])



    def desired_steer(self, t): # [qd, qd dot]
        return [np.sin(t) + 0.7*np.cos(1.5*t)-0.7,
                np.cos(t) - 1.5*0.7*np.sin(1.5*t),
                -np.sin(t) - 1.5*1.5*0.7*np.cos(1.5*t)] # a random path I made up

    def desired_drive(self, t, total_time):
        if t < 0.2*total_time:
            q = 0.5*t
            qdot = 0.5

        elif t < 0.8*total_time:
            q = 0.5*0.2*total_time
            qdot = 0.0
        elif t <= total_time:
            q = -0.5*t+5
            qdot = -0.5
        else:
            q = 0.0
            qdot= 0.0
        return [q, qdot, 0]