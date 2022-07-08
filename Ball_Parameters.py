import numpy as np

# this file will define the ball EOM as presented in the Kayacan paper
class Ball_Robot():
    # define variables that track the balls state
    ball_drive_angle = 0.0  # global drive angle
    pend_drive_angle = 0.0 # pendulum angle w.r.t. vertical measured with the vn
    ball_pipe_angle = np.pi/2 # pipe angle w.r.t. vertical measured with the vn
    steer_angle = 0.0 # steering angle of the pendulum, measured from perpendicular to the pipe

    # ball characteristics
    Ms =
    mp =
    r =
    R =
    Is =
    Ip =

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
    def get_M(self):
        M11 = Ms*R**2 +
        # TODO