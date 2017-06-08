#!/usr/bin/env python
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
import rospy
from exKalmanClass import ExKalman as exkFilter
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32MultiArray, String
from std_msgs.msg import Float32
from mapserver.srv import getMarkPos
import math

pub = rospy.Publisher('kalman_topic', Float32MultiArray, queue_size=1)

filter = exkFilter()
featureDetected = False
featureData = None
get_marking_pos = None

trailerX = trailerY = 0.0 
   

def featureCallback(data):
    global featureDetected
    global get_marking_pos
    global featureData
   
    rospy.loginfo("featuredata")
    lista = str(data).split(';')

    state = filter.getState() 

    x = state.item(0)
    y = state.item(1)
   
    #rospy.loginfo(""+str(float(lista[1])))
   
    res = get_marking_pos(int(float(lista[1])),int(x*1000),int(9000-y*1000))
    x1 = res.x
    y1 = res.y

    if x1 < 0 or y1 < 0:
        featureDetected = False
        featureData = None
    else:
        #rospy.loginfo("x: " + str(x1) + " y: " + str( y1))
        filter.setFeaturePos(x1/1000.0,y1/1000.0) 
        featureData = [float(lista[0][5:]), float(lista[1]), float(lista[2])]
        featureDetected = True
        

def ackermannCallback(data):
    #rospy.loginfo("got ackermann")
    global featureData
    global featureDetected
    global trailerX
    global trailerY
    
    filter.predict([data.steering_angle, data.speed*1.3])
    if featureDetected and featureData:
        filter.update([featureData[0],featureData[1],featureData[2],data.steering_angle, data.speed*1.3], featureDetected)
    else:
        filter.update([0,0,0,data.steering_angle,data.speed],featureDetected)
    featureData = None
    featureDetected = False

    state = filter.getConvertedState()
    mess = Float32MultiArray()
    mess.data = [state.item(0),state.item(1), trailerX, trailerY]
    #rospy.loginfo("theta: " + str(state.item(2)))

    pub.publish(mess)

def trailerCallback(msg):
    global trailerX
    global trailerY
    alpha = msg.data * -1
    l1 = 230
    l2 = 185
    state = filter.getConvertedState()
    x = state.item(0)
    y = state.item(1)
    theta = state.item(2)
    #times 1000 to get meters to mm
    trailerX = x - ((math.cos(theta) * l1) + (math.cos(theta - math.radians(float(alpha))) * l2))
    trailerY = y + ((math.sin(theta) * l1) + (math.sin(theta - math.radians(float(alpha))) * l2))


def runKalman():
    rospy.init_node('exKalmanNode', anonymous=True)
    rospy.Subscriber("master_drive_throttle", AckermannDrive, ackermannCallback, queue_size=1)
    rospy.Subscriber("trailer_sensor", Float32, trailerCallback)

    # setup service handle
    global get_marking_pos
    #rospy.wait_for_service("get_marking_posistion")
    try:
        get_marking_pos = rospy.ServiceProxy('get_marking_position',getMarkPos )
        rospy.loginfo("service init ok..")
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        rospy.loginfo("service init failed")   
    
    
    rospy.Subscriber("feature_detection", String, featureCallback, queue_size=1)
    rospy.spin()


if __name__ == '__main__':
    runKalman()

