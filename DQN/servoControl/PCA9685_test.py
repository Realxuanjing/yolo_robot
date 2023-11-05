import time
import math
import smbus



class PCA9685:
    # Registers/etc.
    __SUBADR1 = 0x02
    __SUBADR2 = 0x03
    __SUBADR3 = 0x04
    __MODE1 = 0x00
    __PRESCALE = 0xFE
    __LED0_ON_L = 0x06
    __LED0_ON_H = 0x07
    __LED0_OFF_L = 0x08
    __LED0_OFF_H = 0x09
    __ALLLED_ON_L = 0xFA
    __ALLLED_ON_H = 0xFB
    __ALLLED_OFF_L = 0xFC
    __ALLLED_OFF_H = 0xFD

    def __init__(self, address=0x40, debug=False):
        self.bus = smbus.SMBus(1)
        self.address = address
        self.debug = debug
        if (self.debug):
            print("Reseting PCA9685")
        self.write(self.__MODE1, 0x00)

    def write(self, reg, value):
        "Writes an 8-bit value to the specified register/address"
        self.bus.write_byte_data(self.address, reg, value)
        if (self.debug):
            # print("I2C: Write 0x%02X to register 0x%02X" % (value, reg))
            pass
    def read(self, reg):
        "Read an unsigned byte from the I2C device"
        result = self.bus.read_byte_data(self.address, reg)
        if (self.debug):
            print("I2C: Device 0x%02X returned 0x%02X from reg 0x%02X" % (self.address, result & 0xFF, reg))
        return result

    def setPWMFreq(self, freq):
        "Sets the PWM frequency"
        prescaleval = 25000000.0  # 25MHz
        prescaleval /= 4096.0  # 12-bit
        prescaleval /= float(freq)
        prescaleval -= 1.0
        if (self.debug):
            print("Setting PWM frequency to %d Hz" % freq)
            print("Estimated pre-scale: %d" % prescaleval)
        prescale = math.floor(prescaleval + 0.5)
        if (self.debug):
            print("Final pre-scale: %d" % prescale)

        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10  # sleep
        self.write(self.__MODE1, newmode)  # go to sleep
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def setPWM(self, channel, on, off):
        "Sets a single PWM channel"
        self.write(self.__LED0_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__LED0_ON_H + 4 * channel, on >> 8)
        self.write(self.__LED0_OFF_L + 4 * channel, off & 0xFF)
        self.write(self.__LED0_OFF_H + 4 * channel, off >> 8)
        if (self.debug):
            # print("channel: %d  LED_ON: %d LED_OFF: %d" % (channel, on, off))
            pass

    def setServoPulse(self, channel, pulse):
        "Sets the Servo Pulse,The PWM frequency must be 50HZ"
        pulse = int(pulse * 4096 / 20000)  # PWM frequency is 50HZ,the period is 20000us
        self.setPWM(channel, 0, pulse)


    def setMotoPluse(self, channel, pulse):
        if pulse > 3000:
            self.setPWM(channel, 0, 3000)
        else:
            self.setPWM(channel, 0, pulse)


pwm = PCA9685(0x40, debug=True)
pwm.setPWMFreq(50)


def set_servo_angle(channel, angle):
    global pwm
    pwm.setServoPulse(channel, 500+angle*2000/180)


calf = [0, 1, 2, 3]

joint = [4, 5, 6, 7]

thigh = [8, 9, 10, 11]

calf_angle = 90
calf_arr = [calf_angle, 180 - calf_angle, calf_angle, 180 - calf_angle]
joint_angle = 90
joint_arr = [joint_angle, 180 - joint_angle, joint_angle, 180 - joint_angle]
thigh_angle = 90
thigh_arr = [thigh_angle, 180 - thigh_angle, thigh_angle, 180 - thigh_angle]


def setting():
    for i in range(4):
        set_servo_angle(calf[i], calf_arr[i])
        set_servo_angle(joint[i], joint_arr[i])
        set_servo_angle(thigh[i], thigh_arr[i])
    time.sleep(0.1)


def setting_calf():
    for i in range(4):
        set_servo_angle(calf[i], calf_arr[i])


def setting_joint():
    for i in range(4):
        set_servo_angle(joint[i], joint_arr[i])


def setting_thigh():
    for i in range(4):
        set_servo_angle(thigh[i], thigh_arr[i])


def setting_improve(first, second, third):
    first()

    second()

    third()
    time.sleep(0.1)


def stand():
    global calf_arr
    global joint_arr
    global thigh_arr
    global calf_angle
    global joint_angle
    global thigh_angle

    calf_angle = 90
    calf_arr = [calf_angle, 180 - calf_angle, calf_angle, 180 - calf_angle]
    joint_angle = 90
    joint_arr = [joint_angle, 180 - joint_angle, joint_angle, 180 - joint_angle]
    thigh_angle = 90
    thigh_arr = [thigh_angle, 180 - thigh_angle, thigh_angle, 180 - thigh_angle]
    setting()



# 5+xiangshang
# 7+#xiangshang
# 4-#xiangshang
# 6-#xiangshang
# 0+xiangshang
# 1-#xiangshang
# 2+#xiangshang
# 3-#xiangshang



def forward():
    global calf_arr
    global joint_arr
    global thigh_arr
    # first and third
    t = 0
    #0
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #1
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [135, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)

    time.sleep(t)
    #2
    calf_arr =  [135, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)

    time.sleep(t)
    #3
    calf_arr =  [90, 90, 135, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 45, 90, 135]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #4
    calf_arr =  [90, 90, 135, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 45, 90, 135]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #5
    calf_arr =  [90, 90, 45, 90]
    joint_arr = [90, 90, 135, 90]
    thigh_arr = [90, 45, 90, 135]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #6
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 45, 90, 135]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #7
    calf_arr =  [90, 90, 90, 135]
    joint_arr = [90, 90, 90, 45]
    thigh_arr = [90, 45, 90, 135]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #8
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 45, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #9
    calf_arr =  [90, 135, 90, 90]
    joint_arr = [90, 45, 90, 90]
    thigh_arr = [90, 45, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #10
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #11
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 45, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #12
    calf_arr =  [90, 45, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #13
    calf_arr =  [90, 90, 90, 45]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [135, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #14
    calf_arr =  [90, 90, 90, 135]
    joint_arr = [90, 90, 90, 45]
    thigh_arr = [135, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #15
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [135, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #16
    calf_arr =  [45, 90, 90, 90]
    joint_arr = [135, 90, 90, 90]
    thigh_arr = [135, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #17
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #18
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 135, 90]
    thigh_arr = [90, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #19
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
def retreat():
    global calf_arr
    global joint_arr
    global thigh_arr
    # first and third
    t = 0
    #0
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #1
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 135, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #2
    calf_arr =  [90, 90, 135, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #3
    calf_arr =  [90, 90, 135, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 135, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #4
    calf_arr =  [90, 90, 135, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 135, 90,45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #5
    calf_arr =  [45, 90, 90, 90]
    joint_arr = [135, 90, 90, 90]
    thigh_arr = [90, 135, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #6
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 135, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #7
    calf_arr =  [90, 135, 90, 90]
    joint_arr = [90, 45, 90, 90]
    thigh_arr = [90, 135, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #8
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #9
    calf_arr =  [90, 90, 90, 135]
    joint_arr = [90, 90, 90, 45]
    thigh_arr = [90, 90, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #10
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #11
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 45]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #12
    calf_arr =  [90, 90, 90, 45]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #13
    calf_arr =  [90, 90, 90, 45]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [45, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #14
    calf_arr =  [90, 135, 90, 90]
    joint_arr = [90, 45, 90, 90]
    thigh_arr = [45, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #15
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [45, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #16
    calf_arr =  [90, 90, 45, 90]
    joint_arr = [90, 90,135, 90]
    thigh_arr = [45, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #17
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [45, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #18
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [135, 90, 90, 90]
    thigh_arr = [45, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #19
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
def move_right():
    global calf_arr
    global joint_arr
    global thigh_arr
    # first and third
    t = 0
    #0
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #1
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 45,90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #2
    calf_arr =  [90, 45, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #3
    calf_arr =  [90, 90, 90, 45]
    joint_arr = [90, 90, 90, 135]
    thigh_arr = [135, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #4
    calf_arr =  [90, 90, 90, 45]
    joint_arr = [90, 90, 90, 135]
    thigh_arr = [135, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #5
    calf_arr =  [90, 90, 90, 135]
    joint_arr = [90, 90, 90, 45]
    thigh_arr = [135, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #6
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [135, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #7
    calf_arr =  [90, 90, 45, 90 ]
    joint_arr = [90, 90, 135, 90]
    thigh_arr = [135, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #8
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [135, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #9
    calf_arr =  [45, 90, 90, 90]
    joint_arr = [135, 90, 90, 90]
    thigh_arr = [135, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #10
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #11
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 135, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #12
    calf_arr =  [90, 90, 135, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #13
    calf_arr =  [135, 90, 90, 90]
    joint_arr = [45, 90, 90, 90]
    thigh_arr = [90, 135, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #14
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [135, 90, 90, 90]
    thigh_arr = [90, 135, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #15
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 135, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #16
    calf_arr =  [90, 135, 90, 90]
    joint_arr = [90, 45, 90, 90]
    thigh_arr = [90, 135, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #17
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #18
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 45]
    thigh_arr = [90, 90, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #19
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
def move_left():
    global calf_arr
    global joint_arr
    global thigh_arr
    # first and third
    t = 0
    #0
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #1
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90,90, 45]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #2
    calf_arr =  [90, 90, 90, 45]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #3
    calf_arr =  [90, 45, 90, 90]
    joint_arr = [90, 135, 90, 90]
    thigh_arr = [45, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #4
    calf_arr =  [90, 45, 90, 90]
    joint_arr = [90, 135, 90, 90]
    thigh_arr = [45, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #5
    calf_arr =  [90, 135, 90, 90]
    joint_arr = [90, 45, 90, 90]
    thigh_arr = [45, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #6
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [45, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #7
    calf_arr =  [45, 90,90, 90 ]
    joint_arr = [135, 90, 90, 90]
    thigh_arr = [45, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #8
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #9
    calf_arr =  [90, 90, 45, 90]
    joint_arr = [90, 90, 135, 90]
    thigh_arr = [90, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #10
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #11
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [135, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #12
    calf_arr =  [135, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #13
    calf_arr =  [90, 90, 135, 90]
    joint_arr = [90, 90, 45, 90]
    thigh_arr = [90, 45, 90, 135]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    #14
    calf_arr =  [90, 90, 45, 90]
    joint_arr = [90, 90, 135, 90]
    thigh_arr = [90, 45, 90, 135]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #15
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 45, 90, 135]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #16
    calf_arr =  [90, 90, 90, 135]
    joint_arr = [90, 90, 90, 45]
    thigh_arr = [90, 45, 90, 135]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #17
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 45, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #18
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 45, 90, 90]
    thigh_arr = [90, 45, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    #19
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)

def turn_left():
    global calf_arr
    global joint_arr
    global thigh_arr
    # first and third
    t = 0
    #0
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [135, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [135, 90, 90, 90]
    thigh_arr = [135, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [135, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 135, 90]
    thigh_arr = [135, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 135, 90]
    thigh_arr = [135, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90 ,90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [135, 90, 135, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90 ,90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90,45,90,45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90 ,90]
    joint_arr = [90, 45, 90, 90]
    thigh_arr = [90,45,90,45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90 ,90]
    joint_arr = [90, 45, 90, 90]
    thigh_arr = [90,90,90,45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90 ,90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90,90,90,45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90 ,90]
    joint_arr = [90, 90, 90, 45]
    thigh_arr = [90,90,90,45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90 ,90]
    joint_arr = [90, 90, 90, 45]
    thigh_arr = [90,90,90,90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90 ,90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90,90,90,90]
    setting_improve(setting_joint, setting_calf, setting_thigh)


def turn_right():
    global calf_arr
    global joint_arr
    global thigh_arr
    # first and third
    t = 0
    #0
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 45, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 45, 90, 90]
    thigh_arr = [90, 45, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 45, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 45]
    thigh_arr = [90, 45, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 45]
    thigh_arr = [90, 45, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 45, 90, 45]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [45, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [135, 90, 90, 90]
    thigh_arr = [45, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [135, 90, 90, 90]
    thigh_arr = [90, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 135, 90]
    thigh_arr = [90, 90, 45, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 135, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)
    calf_arr =  [90, 90, 90, 90]
    joint_arr = [90, 90, 90, 90]
    thigh_arr = [90, 90, 90, 90]
    setting_improve(setting_joint, setting_calf, setting_thigh)
    time.sleep(t)




if __name__ == '__main__':
    print("start the control")
    stand()
    time.sleep(2)
    move_left()

