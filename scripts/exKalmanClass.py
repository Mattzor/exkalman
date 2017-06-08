"""
Copyright (c) 2017, Robert Krook
Copyright (c) 2017, Erik Almblad
Copyright (c) 2017, Hawre Aziz
Copyright (c) 2017, Alexander Branzell
Copyright (c) 2017, Mattias Eriksson
Copyright (c) 2017, Carl Hjerpe
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Chalmers University of Technology nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
#import matplotlib.dates as mdates
#import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from sympy import Symbol, symbols, Matrix, sin, cos, tan
from sympy import init_printing



class ExKalman(object):
    def __init__ (self):  

	self.truckAngle = 0
        self.markAngle = 0

        self.numstates = 4 # States
        self.dt = 1.0/20.0 # Sample Rate of the Measurements is 30Hz
        self.w = 0.24      # Wheelbase

        self.featureX = 0.0
        self.featureY = 0.0
        self.xs, self.ys, self.vs, self.thetas, self.betas, self.dts, self.ws, self.rs, self.alphas = symbols('x y v theta beta dt w R alpha')
        self.rs = self.ws/tan(self.alphas)
        self.betas = ((self.vs*self.dts)/self.ws)*tan(self.alphas)

        self.gs = Matrix([[self.xs+(self.rs*self.vs)*(sin(self.thetas+self.betas)-sin(self.thetas))],
                 [self.ys+(self.rs*self.vs)*(cos(self.thetas)-cos(self.thetas+self.betas))],
                 [self.thetas + self.betas],
                 [self.vs]])
        self.state = Matrix([self.xs, self.ys, self.thetas, self.vs])
        self.F = self.gs.jacobian(self.state)  # State transition matrix
        self.const_vel = 0.5

        #  uncertainty---------------------------------------------------------------------------------------
        self.P = np.diag([0.0, 0.0, 0.0, 0.0])

        # Process Noise Covariance Matrix Q---------------------------------------------------------------------------
        self.sCourse  = 0.01*self.dt # assume 0.1rad/s as maximum turn rate for the vehicle
        self.sVelocity= 0.01*self.dt # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
        self.Q = np.diag([0, 0, self.sCourse**2, self.sVelocity**2])

        #  state -------------------------------------------------------------------------------------------------------
        x_start = 1005.0/1000.0   # 943.0/1000.0
        y_start = (9000.0-4140.0)/1000.0
        dir_start =  np.pi/2.0
        vel_start = 0.0
        self.x = np.matrix([[x_start, y_start, dir_start, vel_start]]).T
        self.U=float(np.cos(self.x[2])*self.x[3])
        self.V=float(np.sin(self.x[2])*self.x[3])        
        self.hs = Matrix([[self.thetas],[self.vs]])        
        self.JHs=self.hs.jacobian(self.state)
        # Measurement noise covariance R ------------------------------------------------------------------------
        vardist  = 0.0000001
        varphi   = 0.0000001
        vartheta = 0.0000001
        varspeed = 0.01*self.dt  # Variance of the speed measurement
        varang   = 0.1*self.dt     # Variance of the yawrate measurement
        self.Rcov = np.matrix([[vardist**2, 0.0, 0.0, 0.0, 0.0],
                             [0.0, varphi**2, 0.0, 0.0, 0.0],
                             [0.0, 0.0, vartheta**2, 0.0, 0.0],
                             [0.0, 0.0, 0.0, varang**2, 0.0],
                             [0.0, 0.0, 0.0 ,0.0, varspeed**2]])

        # Identity matrix -------------------------------------------------------------------------------------------------------
        self.I = np.eye(self.numstates)

    # Measurement function H 
    # ======================
    # Predicted measurement from predicted state?
    # def H(self):
    #    self.JHs=self.hs.jacobian(self.state)



    # Time Update (Prediction)          
    # ========================          
    # Project the state ahead    
    def predict(self,u):      

        u[0] = u[0]/180.0*np.pi
        
        alpha = u[0]        
        current_speed = u[1]

        if  np.abs(alpha) < 0.01 : # Driving straight
            self.x[0] = self.x[0] + self.x[3]*self.dt * np.cos(self.x[2])
            self.x[1] = self.x[1] + self.x[3]*self.dt * np.sin(self.x[2])
            self.x[2] = self.x[2]
            self.x[3] = current_speed
        else: # otherwise (implement dynamic matrix)
            self.B = float(current_speed*self.dt/self.w*tan(alpha))
            self.R = float(self.w/(np.tan(alpha)+0.00001))
            self.x[0] = self.x[0] + self.R * (np.sin(self.x[2]+self.B) - np.sin(self.x[2]))
            self.x[1] = self.x[1] + self.R * (-np.cos(self.x[2]+self.B)+ np.cos(self.x[2]))
            self.x[2] = self.x[2] + self.B
            self.x[3] = current_speed
        
        # Calculate the Jacobian of the Dynamic Matrix A
        # see "Calculate the Jacobian of the Dynamic Matrix with respect to the state vector"
        self.B = float(self.x[3]*self.dt/self.w*tan(self.x[2]))
        self.R = float(self.w/(np.tan(self.x[2])+0.00001))
        a13 = float(self.R * (np.cos(self.B+self.x[2]) - np.cos(self.x[2])))
        a14 = float(self.dt *cos(self.B+self.x[2]))
        a23 = float(self.R * (np.sin(self.B+self.x[2]) - np.sin(self.x[2])))
        a24 = float(self.dt * sin(self.B+self.x[2]))
        a34 = float((self.dt/self.w) * tan(alpha))
        self.JA = np.matrix([[1.0, 0.0, a13, a14],
                        [0.0, 1.0, a23, a24],
                        [0.0, 0.0, 1.0, a34],
                        [0.0, 0.0, 0.0, 1.0]])
        
        # Project the error covariance ahead
        self.P = self.JA*self.P*self.JA.T + self.Q           # Q = process noise covar
        
    # Measurement Update (Correction)
    # ===============================
    # Measurement Function
    # Measurement argument should be [angle, speed] now as we use same data from of
    # Feature argument should be []
    def update(self, measurement, feature):

	# measurement[0] = distance   from feature
        # measurement[1] = globalangle from feature
        # measurement[2] = markangle   from feature
        # measurement[3] = ackermann steering angle
        # measurement[4] = ackermann speed


        # Convert all angle from degrees to radians
        measurement[1] = measurement[1]/180.0*np.pi    
        measurement[2] = measurement[2]/180.0*np.pi
        measurement[3] = measurement[3]/180.0*np.pi
        

        self.hx = np.matrix([[float(self.x[0])],       # Predicted x
                             [float(self.x[1])],       # Predicted y
                             [float(self.x[2])],       # Predicted global angle
                             [float(measurement[3])],  # Ackermann steeringangle
                             [float(self.x[3])]        # Predicted speed
                            ])

        if feature:
            self.JH = np.matrix([[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0]]) 
        else:
            self.JH = np.matrix([[0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]]) 


        self.S = self.JH*self.P*self.JH.T + self.Rcov
        self.K = (self.P*self.JH.T) * np.linalg.inv(self.S)

        # Update the estimate 
        
        self.Z = np.array(measurement).reshape(self.JH.shape[0],1)     

        if feature:
           
            self.markAngle = self.Z.item(2)
	    self.truckAngle = self.Z.item(1)
            
            Z0 = float(self.Z.item(0)) / 10.0
            Z1 = float(self.Z.item(1))
            print("=========================\n")
            print("featureX: " + str(self.featureX))
            print("featureY: " + str(self.featureY))
            print("X before: " + str(self.x.item(0)))
            print("Y before: " + str(9-self.x.item(1)))
            print("=========================\n")

            self.Z[0] = self.featureX - Z0 * cos( 2*np.pi - (self.Z.item(1)+self.Z.item(2) ))
            self.Z[1] = 9 - self.featureY + Z0 * sin( 2*np.pi - (self.Z.item(1)+self.Z.item(2)) )
            self.Z[2] = Z1
            print("######## after #######")
            print("X measured: " + str(self.Z[0]))
            print("Y measured: " + str(9 - self.Z[1]))
            print("######## END #######")
           
 
        self.y = self.Z - (self.hx)                         # Innovation or Residual
        self.x = self.x + (self.K*self.y)

        # Update the error covariance
        self.P = (self.I - (self.K*self.JH))*self.P


    def getFeaturePos(self):
        theta = self.x.item(2)
        x = self.x.item(0)
        y = self.x.item(1)
        return [self.serviceHandle(theta,x,y)]    

    def getState(self):
        return self.x

    def getAngle(self):
        return self.theta

    def setFeaturePos(self,x,y):
	self.featureX = x
        self.featureY = y

    def getConvertedState(self):
        x = self.x[0] * 1000
        y = 9000 - self.x[1]*1000
        return np.array([x,y,self.x[2],self.x[3]])


#if __name__ == '__main__':
    #filter.H()
    #filter = ExKalman()
    #filter.predict([90, 10])
    #filter.update([90, 10])
#    print(filter.getState())
    #filter.H()
    #filter.predict([93, 10])
    #filter.update([93, 10])
#    print(filter.getState())
    #filter.H()
    #filter.predict([92, 10])
    #filter.update([92, 10])
    #print(filter.getState())











