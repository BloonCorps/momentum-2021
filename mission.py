import asyncio #asynchronous stuff
from navpy.core.navpy import lla2ned #conversion between lat-long-alt and north-east-down (x, y, z) 
import pygazebo #helps get info from gazebo
import pygazebo.msg.v11.laserscan_stamped_pb2 #helps get lidar info from gazebo
import numpy as np # for transposing across coordinate systems, AKA hardcore matrix operations
from mavsdk import System #mavsdk is the python code we write
from mavsdk.mission import (MissionItem, MissionPlan) #missio nsetting stuff
import navpy #coordinate conversions between lat-long-alt and north-east-down (x, y, z) 
import time #prevents spamming

# This is the gazebo master address and port from PX4 message `[Msg] Connected to gazebo master @ http://127.0.0.1:11345`
HOST, PORT = "127.0.0.1", 11345

##################################################
### LIDAR GATHERING AND QUATERNION OPERATIONS  ###
##################################################
class GazeboMessageSubscriber: 
    '''
    Gazebo Message Subscriber subscribes to the lidar messages published by Gazebo.
    Explore the lidar tutorial in this repository to understand how it works.
    '''
    def __init__(self, host, port, timeout=30):
        self.host = host
        self.port = port
        self.loop = asyncio.get_event_loop()
        self.running = False
        self.timeout = timeout

    async def connect(self):
        connected = False
        for i in range(self.timeout):
            try:
                self.manager = await pygazebo.connect((self.host, self.port))
                connected = True
                break
            except Exception as e:
                print(e)
            await asyncio.sleep(1)

        if connected: 
            # info from gz topic -l, gz topic -i arg goes here
            self.gps_subscriber = self.manager.subscribe('/gazebo/default/iris_lmlidar/lmlidar/link/lmlidar/scan', 'gazebo.msgs.LaserScanStamped', self.LaserScanStamped_callback)

            await self.gps_subscriber.wait_for_connection()
            self.running = True
            while self.running:
                await asyncio.sleep(0.1)
        else:
            raise Exception("Timeout connecting to Gazebo.")

    def LaserScanStamped_callback(self, data):
        # What *_pb2 you use here depends on the message type you are subscribing to
        self.LaserScanStamped = pygazebo.msg.v11.laserscan_stamped_pb2.LaserScanStamped()
        self.LaserScanStamped.ParseFromString(data)
    
    async def get_LaserScanStamped(self):
        for i in range(self.timeout):
            try:
                return self.LaserScanStamped
            except Exception as e:
                pass
            await asyncio.sleep(1)


def quat_mult(quaternion1, quaternion0):
    # Hamilton Product
    w0 = quaternion0[0]
    x0 = quaternion0[1]
    y0 = quaternion0[2]
    z0 = quaternion0[3]
    w1 = quaternion1[0]
    x1 = quaternion1[1]
    y1 = quaternion1[2]
    z1 = quaternion1[3]
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def get_world_obst(lsr_val):
    # Laser returns are returned in a 1D array ordered as [v1h1 .. v20h1 v1h2 .. v20h2 v1h3 .. v20h3 ...], where v is the vertical index, and h is the horizontal index.
    # https://math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion    
    # All quaternions expressed in the order [w, i, j, k]
    # Rotation quat
    R       = np.array([lsr_val.scan.world_pose.orientation.w,  lsr_val.scan.world_pose.orientation.x,  lsr_val.scan.world_pose.orientation.y,  lsr_val.scan.world_pose.orientation.z])

    # Conjugate of rotation quat
    R_prime = np.array([lsr_val.scan.world_pose.orientation.w, -lsr_val.scan.world_pose.orientation.x, -lsr_val.scan.world_pose.orientation.y, -lsr_val.scan.world_pose.orientation.z])

    # Linear displacement vector in pure quat form (0.1i is the translation from sensor frame to vehicle frame)
    disp    = np.array([[0], [lsr_val.scan.world_pose.position.x + 0.1], [lsr_val.scan.world_pose.position.y], [lsr_val.scan.world_pose.position.z]])
    print(disp)

    # Linearly space vector describing horizontal range in radians
    h_range = np.linspace((lsr_val.scan.angle_min), (lsr_val.scan.angle_max), lsr_val.scan.count, dtype=np.float64)

    # Linearly space vector describing vertical range in radians
    v_range = np.linspace((lsr_val.scan.vertical_angle_min*-1.0 + np.pi/2.0), (lsr_val.scan.vertical_angle_max*-1.0 + np.pi/2.0), lsr_val.scan.vertical_count, dtype=np.float64)

    # Tile and repeat variables to allow vectorized operations
    R       = np.tile(R,          lsr_val.scan.count * lsr_val.scan.vertical_count)
    R_prime = np.tile(R_prime,    lsr_val.scan.count * lsr_val.scan.vertical_count)  
    disp    = np.tile(disp,       lsr_val.scan.count * lsr_val.scan.vertical_count)
    h_range = np.tile(h_range,    lsr_val.scan.vertical_count)
    v_range = np.repeat(v_range,  lsr_val.scan.count)

    # Get x, y, z components of the range readings in sensor frame and pad to assume pure quat form
    # https://tutorial.math.lamar.edu/classes/calciii/SphericalCoords.aspx
    P = np.array([lsr_val.scan.ranges * np.sin(v_range) * np.cos(h_range), lsr_val.scan.ranges * np.sin(v_range) * np.sin(h_range), lsr_val.scan.ranges * np.cos(v_range)])

    # Double check that the equality still holds
    # print(np.equal(np.round(np.square(lsr_val.scan.ranges), 4), np.round(np.square(P[0]) + np.square(P[1]) + np.square(P[2]), 4)))
    
    P = np.concatenate((np.array([np.zeros(P.shape[1])]), P), axis=0)

    # Multiply out the rotation (RPR') and add vehicle + sensor offsets
    P_prime = quat_mult(quat_mult(R, P), R_prime) + disp

    # Remove the pad and transpose for easy reading 
    result = P_prime[1:4].T

    # Access using the index of the return you'd like (in the format[v1h1 .. v20h1 v1h2 .. v20h2 v1h3 .. v20h3 ...]) and index of axis x=0, y=1, z=2
    # e.g. result[-60:-40, 0:2] gives 20 x-y-z values starting from 60 from the end and ending at 40 from the end

    return result

##################################################
### MISSION CODE STUFF                         ###
##################################################

async def observe_is_in_air(drone, running_tasks):
    """ Monitors whether the drone is flying or not and
    returns after landing """

    print("Observing in air")

    was_in_air = False

    async for is_in_air in drone.telemetry.in_air():
        if is_in_air:
            was_in_air = is_in_air

        if was_in_air and not is_in_air:
            for task in running_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await asyncio.get_event_loop().shutdown_asyncgens()

            return

def get_next_wp (lsr_val, lla_ref_off):
    '''
    Generates waypoints given lidar-data. lla_ref_off is deprecated.
    '''
    max_speed = 12 # m/s
    #The waypoints are in (x, y, z), the same readings given by the lidar world position in NED
    #lla_ref_off was bugged (giving values of [-4621324.0837164, -947769.59003018, 2097587.87792365])
    #So we let lla_ref_off be [0, 0, 0]

    #lla_ref_off = navpy.lla2ned(0, 0, 0, lla_ref[0], lla_ref[1], lla_ref[2])
    lla_ref_off = [0, 0, 0]

    own_x = lsr_val.scan.world_pose.position.x + lla_ref_off[1]

    #Transforms lidar data into obstacles that we can use
    registered_scans = get_world_obst(lsr_val) 
    
    spam_terminal = True #on and off switch to have it spam waypoints or not.
    start = time.time()
    current_time = 0
    last_time = 0
    while (current_time < 1e20):
        current_time = round(time.time() - start)
        if((current_time % 1 == 0 and last_time < current_time) or spam_terminal): 
            last_time = current_time

            filtered_scans = filter_scans(registered_scans, own_x)

            slope = filtered_scans[1]
            good_scans = filtered_scans[0]

            slope = slope/5 #our program over estimates slopes because of the inherent variance in the data 
            
            #target AGL is 3, and y = 0, are hardcoded values given by LM
            AGL = 3
            y = 0
            look_ahead = 0 #option to look ahead and use slope prediction

            waypoint = good_scans[0] #option to go to closest point
            #waypoint = good_scans[int(len(good_scans)/2)] #option to go to midpoint
            #waypoint = good_scans[len(good_scans)-1] #option to go to furthest point

            if own_x > 34:
                x = float("Nan")
                y = float("Nan")
                z = float("Nan")
            else:
                x = waypoint[0] + look_ahead
                z = waypoint[2] + AGL + slope*look_ahead

            return [x, y, z, max_speed]

def filter_scans(registered_scans, own_x):
    '''
    Output: calculates slope of terrain
    and finds waypoints along the path we travel
    '''
    temp = []

    for item in registered_scans:
        if ((item[1] < 1.15) and (item[1] > 0.85)):
            temp.append(item) 

    sorted(temp, key=lambda terrain: terrain[0])

    mindex = 0 #find_min(temp)
    max_index = len(temp) - 1 #find_max(temp)

    z_one = temp[max_index][2]
    z_zero = temp[mindex][2]
    x_one = temp[max_index][0]
    x_zero = temp[mindex][0]    

    slope = (z_one - z_zero)/(x_one - x_zero)

    return [temp, slope]

"""
Finding max and min terrain waypoint functions. Not used now but kept for reference.
"""
def find_min(scanned, axis = 2): #axis of 0 indicates x, 1 indicates y, and 2 indicates z
    min = 1e20
    mindex = None
    for index in range(0, len(scanned), 1):
        if scanned[index][axis] < min:
            min = scanned[index][axis]
            mindex = index
    return mindex

def find_max(scanned, axis = 2):
    max = -1e20
    max_index = None
    for index in range(0, len(scanned), 1):
        if scanned[index][axis] > max:
            max = scanned[index][axis]
            max_index = index
    return max_index

#########################################################################
###  MAIN RUNNING FUNCTION                                            ###
#########################################################################
async def run_mission(drone, mission_items, lla_ref, gz_sub):
    #Keep in mind we do not account for the fact that the thing is slightly off   
    end_point = [39.0018754041018363, 0.9619961870448452, 0.91699984390288591] 

    #The ned values are slightly skewed because the lla reference is at the start location, not world origin
    #lla_ref_off = navpy.lla2ned(0, 0, 0, lla_ref[0], lla_ref[1], lla_ref[2])
    #lla_ref_off is in NED coordinates
    #LLA_REF_OFF = [-4621324.0837164, -947769.59003018, 2097587.87792365] 
    #LLA_REF = [47.3977508, 8.5456073, 488.02301025390625]

    #making lla_ref_off like this seems to work the best
    lla_ref_off = [0, 0, 0]
    [lat, lon, alt] = lla_ref # Set default mission item to start location
    
    print("-- Comencing mission")

    while(True):
        lsr_val = await gz_sub.get_LaserScanStamped()

        [x, y, z, speed] = get_next_wp(lsr_val, lla_ref_off)

        if(not np.isnan(x) or x < 34): #near the end
            #If the x coordinate of the waypoint is a number (aka not-not a number (fuck double negatives)

            [lat, lon, alt] = navpy.ned2lla([y, x, -z], lla_ref[0], lla_ref[1], lla_ref[2])
            await drone.action.set_maximum_speed(speed)
            await drone.action.goto_location(lat, lon, alt, 90)

        else:

            own_x = lsr_val.scan.world_pose.position.x + lla_ref_off[1]
            own_z = lsr_val.scan.world_pose.position.z - lla_ref_off[2]
            print("-- Going to the end")
            # print("-- Mission is done")
            await drone.action.set_maximum_speed(12)
            [lat, lon, alt] = navpy.ned2lla([end_point[1], end_point[0], -end_point[2]], lla_ref[0], lla_ref[1], lla_ref[2])
            if(own_x >= end_point[0] - 5 and abs(own_z - end_point[2]) <= 0.5):
                print("-- landing")
                await drone.action.land()
                break
            else:
                await drone.action.goto_location(lat, lon, alt, 90)            

async def run():
    # Some of the values are inf and numpy complains when doing math with them, so turn off warnings
    np.seterr(all='ignore') 
    '''
    Code below gathers lidar-data
    '''
    # Make the Gazebo subscriber object 
    gz_sub = GazeboMessageSubscriber(HOST, PORT)
    # Actually do the subscription
    gz_task = asyncio.ensure_future(gz_sub.connect()) #create_task -> ensure_future
    # Get a test message - this will speed up all of the subsequent requests
    await gz_sub.get_LaserScanStamped()

    # Initialize drone and connect
    drone = System()
    await drone.connect(system_address="udp://:14540")
    """
    Block of print statements to check everything is ok.
    """
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Drone discovered with UUID: {state.uuid}")
            break
    # Make sure telemetry is good so that we know we can trust the position and orientation values
    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            print("Global position estimate ok")
            break
    # Get our starting position and use that as reference for conversions between lla and ned
    print("Fetching amsl altitude at home location....")
    lla_ref = []
    async for terrain_info in drone.telemetry.position():
        lla_ref = [terrain_info.latitude_deg, terrain_info.longitude_deg, terrain_info.absolute_altitude_m]
        break
    
    # start = navpy.lla2ned(lla_ref[0], lla_ref[1], lla_ref[2], 0, 0, 0)
    # start = [1.0018754041018363, 0.9619961870448452, -0.08300015609711409]
    # [lla_ref[0], lla_ref[1], lla_ref[2] + 3, 90] = [47.3977509, 8.5456072, 490.96502685546875, 90]

    # First mission item is to takeoff to just above our starting location
    await drone.action.set_maximum_speed(12)
   
    await drone.action.goto_location(lla_ref[0], lla_ref[1], lla_ref[2] + 3, 90)

    # Last mission item is at the end of the terrain, just above the terrain
    # All of your mission items go between the the start point above and the end point below
    
    # Spool up the mission and mission monitor
    mission_task = asyncio.ensure_future(run_mission(drone, [], lla_ref, gz_sub))
    termination_task = asyncio.ensure_future(observe_is_in_air(drone, [mission_task, gz_task]))

    # Make the drone dangerous (arm it)
    print("-- Arming")
    await drone.action.arm()
    await asyncio.sleep(1)

    await drone.action.set_takeoff_altitude(1.5)
    await drone.action.takeoff()

    await asyncio.sleep(1)

    await termination_task


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
