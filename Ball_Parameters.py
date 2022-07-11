import numpy as np

# this file will define the ball EOM as presented in the Kayacan paper
class Ball_Robot():
    # ball characteristics shared by all instances
    Ms =  0.0
    mp = 0.0
    r = 0.0
    R = 0.0
    Is = 0.0
    Ip = 0.0

    def __init__(self):# define variables that track the balls state unique to this instance
        self.ball_drive_angle = 0.0  # global drive angle
        self.pend_drive_angle = 0.0  # pendulum angle w.r.t. vertical measured with the vn
        self.ball_pipe_angle = np.pi / 2  # pipe angle w.r.t. vertical measured with the vn
        self.steer_angle = 0.0  # steering angle of the pendulum, measured from perpendicular to the pipe

    # define some updaters
    def update_ball_drive(self,x):
        self.ball_drive_angle = x
    def update_pend_drive(self, x):
        self.pend_drive_angle = x
    def update_pipe_angle(self, x):
        self.ball_pipe_angle = x
    def update_steer(self, x):
        self.steer_angle = x

    # define methods to output ball state mtx's
    def get_M(self, q): #[sphere driving, pendulum drive, pipe angle, steering angle]
        q1, q2, q3, q4 = q
        M11 = self.Ms*self.R**2 + self.mp*self.R**2 +self.mp*self.r**2 + self.Is + self.Ip + 2*self.mp*self.R*self.r*np.cos(q2-q1)
        M12, M21 = -self.mp*self.r**2 - self.Ip - self.mp*self.R*self.r*np.cos(q2-q1)
        M13, M14, M23, M24, M31, M32, M41, M42 = 0
        M22 = self.mp*self.r**2 + self.Ip
        M33 = self.Ms*self.R**2 + self.mp*self.R**2 + self.mp*self.r**2 + self.Is + self.Ip + 2*self.mp*self.R*self.r*np.cos(q3-q4)
        M34, M43 = -self.mp*self.r**2 - self.Ip - self.mp*self.R*self.r*np.cos(q4-q3)
        M44 = self.mp*self.r**2 + self.Ip

        return np.array([[M11, M12, M13, M14],
                         [M21, M22, M23, M24],
                         [M31, M32, M33, M34],
                         [M41, M42, M43, M44]])

    def get_V(self, q):  # [sphere driving, pendulum drive, pipe angle, steering angle]

        q1, q2, q3, q4 = q
        V11 = self.mp*self.R*self.l*np.sin(q2-q1)*(q1-q2)**2 - self.mp*self.g*self.r*np.sin(q2-q1)
        V21 = self.mp*self.g*self.r*np.sin(q2-q1)
        V31 = self.mp * self.R * self.l * np.sin(q4 - q3) * (q4 - q3) ** 2 - self.mp * self.g * self.r * np.sin(q4 - q3)
        V41 = self.mp * self.g * self.r * np.sin(q4 - q3)
        return np.array([[V11],
                        [V21],
                        [V31],
                        [V41]])