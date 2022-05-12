#!/usr/bin/env python

import rospy
from sound_play.libsoundplay import SoundClient
from sound_play.msg import SoundRequest

# Author: Ronaldson Bellande

class robot_speech(object):
    def __init__(self):
        pass


def robot_speaking(speech, volume, sleep):
    
    rospy.loginfo("Robot Speaking")
    soundhandle = SoundClient()
    rospy.sleep(sleep)

    speech.play(str(speech) + ".wav", volume=volume)
    rospy.sleep(sleep)

    soundhandle.play(SoundRequest.NEEDS_UNPLUGGING, blocking=True)



def robot_speaking_say(speech, volume, sleep):
    
    rospy.loginfo("Robot Speaking")
    soundhandle = SoundClient()
    rospy.sleep(sleep)
    
    soundhandle.say(str(speech))
    rospy.sleep(sleep)
    
    soundhandle.play(SoundRequest.NEEDS_UNPLUGGING, blocking=True)


if __name__ == '__main__':
    rospy.init_node("Robot Speaking", anonymous=False)
    robot_speacking()
    robot_speaking_say()
    rospy.loginfo('Finished')
