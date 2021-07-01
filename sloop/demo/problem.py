import pomdp_py
import airsim
import spatial_foref.oopomdp.problem as mos
from spatial_foref.oopomdp.env.env import MosEnvironment
from .utils import quat_to_euler, to_radians, to_degrees
from spatial_foref.oopomdp.env.env import MosEnvironment
from spatial_foref.oopomdp.domain.action import *
from spatial_foref.oopomdp.models.transition_model import RobotTransitionModel
from spatial_foref.oopomdp.domain.state import *
import copy

class AirSimSearchProblem(pomdp_py.POMDP):

    def __init__(self, mapinfo, map_name, agent, env, name="AirSimSearch"):
        super().__init__(agent, env, name=name)
        self.mapinfo = mapinfo
        self.map_name = map_name

    @classmethod
    def from_mos_problem(cls, mos_problem, mapinfo,
                         init_point_pair, second_point_pair, flight_height,
                         map_name="neighborhoods",
                         landmark_heights={}):
        """
        Given a mos_problem (MosOOPOMDP), convert it to the
        AirSimSearchProblem that we want here. Essentially
        changing the environment.
        """
        print("Converting to MosOOPOMDP to an AirSim Problem")
        RobotTransitionModel.MAPINFO = mapinfo
        RobotTransitionModel.MAPNAME = map_name
        RobotTransitionModel.FLIGHT_HEIGHT = flight_height
        RobotTransitionModel.LANDMARK_HEIGHTS = landmark_heights
        env = AirSimSearchEnvironment.from_mos_env(
            mos_problem.env,init_point_pair, second_point_pair, flight_height)
        agent = mos_problem.agent
        return AirSimSearchProblem(mapinfo, map_name, agent, env)


class AirSimSearchEnvironment(MosEnvironment):
    # pomdp Environment for AirSim search.
    def __init__(self, init_point_pair, second_point_pair, height, *args, **kwargs):
        """
        Args:
            init_point_pair: A pair of two points, matching 2D points between UE4 and POMDP
                for the drone's starting location.
            second_point_pair: Another pair of two points. Could be at other places.
            height: Height that the drone should fly at (NED system)
            init_ue4_loc: Initial UE4 2D coordinate (on the x-y plane)
        """
        super().__init__(*args, **kwargs)

        # Figure out the resolution
        pomdp_loc1, ue4_loc1 = init_point_pair
        pomdp_loc2, ue4_loc2 = second_point_pair

        pomdp_diffx = pomdp_loc2[0] - pomdp_loc1[0]
        pomdp_diffy = pomdp_loc2[1] - pomdp_loc1[1]

        ue4_diffx = ue4_loc2[0] - ue4_loc1[0]
        ue4_diffy = ue4_loc2[1] - ue4_loc1[1]

        # ue4 displacement per pomdp grid cell
        self.res_x = ue4_diffx / pomdp_diffx
        self.res_y = ue4_diffy / pomdp_diffy
        self.init_ue4_pos = ue4_loc1
        self.init_pomdp_pos = pomdp_loc1
        self.height = -abs(height)

        # connect to airsim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # make drone hover at height
        self.client.moveToZAsync(self.height, 1).join()

        # Obtain the groudtruth robot state
        robot_id = list(self.robot_ids)[0]
        ned_pose = self.ned_pose()
        next_rx, next_ry, next_rth = self.ned_to_pose2d(ned_pose)
        self.state.object_states[robot_id] = RobotState(robot_id,
                                                        (next_rx, next_ry, next_rth),
                                                        tuple(),
                                                        action.name)
        print(self.state.object_states[robot_id])


    @classmethod
    def from_mos_env(cls, mos_env, init_point_pair, second_point_pair, height):
        return AirSimSearchEnvironment(init_point_pair, second_point_pair, height,
                                       (mos_env.width, mos_env.length),
                                       mos_env.state,
                                       mos_env.sensors,
                                       mos_env.obstacles,
                                       no_look=mos_env.no_look,
                                       reward_small=mos_env.reward_model.small)



    def ned_pose(self, quat=True):
        state = self.client.getMultirotorState()
        _pos = state.kinematics_estimated.position
        pos = [_pos.x_val, _pos.y_val, _pos.z_val]
        _orien = state.kinematics_estimated.orientation
        orien = [_orien.x_val, _orien.y_val, _orien.z_val, _orien.w_val]
        if not quat:
            orien = quat_to_euler(*orien)
        return {"position": pos,
                "orientation": orien}

    def ned_to_pose2d(self, ned_pose):
        """Convert NED to pomdp grid + yaw"""
        # First convert position to UE4's; first get the relative pose then
        # account for the UE4 origin......
        # NOTE... for some reason to me UE4 and NED axes are aligned
        ned_x, ned_y, _ = ned_pose["position"]
        ue4_rel_x = ned_x * 100.0   # meter to centimeters
        ue4_rel_y = ned_y * 100.0

        # Convert ue4 pose to pomdp grid
        pomdp_rel_x = ue4_rel_x / self.res_x
        pomdp_rel_y = ue4_rel_y / self.res_y
        pomdp_x = int(round(self.init_pomdp_pos[0] + pomdp_rel_x))
        pomdp_y = int(round(self.init_pomdp_pos[1] + pomdp_rel_y))

        # Then convert orientation. First convert quaternion
        # to Euclidean.
        qx, qy, qz, qw = ned_pose["orientation"]
        thx, thy, thz = quat_to_euler(qx, qy, qz, qw)
        # +180 to account for angular transform between POMDP grid (0 deg at +x)
        # with NED (0 deg at -x).
        return (pomdp_x, pomdp_y, to_radians(thz))

    def pose2d_to_ned(self, pomdp_x, pomdp_y):
        """Returns a 2D ned position"""
        ue4_rel_x = (pomdp_x - self.init_pomdp_pos[0]) * self.res_x
        ue4_rel_y = (pomdp_y - self.init_pomdp_pos[1]) * self.res_y
        ned_x = (ue4_rel_x) / 100.0
        ned_y  = (ue4_rel_y) / 100.0
        return (ned_x, ned_y)

    def state_transition(self, action, execute=True, robot_id=None):
        """State transition of AirSim.
        Executes the action; if it is a move action, then control the
        drone in AirSim to execute that control. After the control
        action is finished, update the robot state to be what is
        obtained from AirSim client."""

        next_state = self.transition_model.sample(self.state, action)
        if not execute:
            reward = self.reward_model.sample(state, action, next_state, robot_id=robot_id)
            return next_state, reward

        robot_state = next_state.object_states[robot_id]
        if isinstance(action, MotionAction):
            robot_pose = next_state.object_states[robot_id].pose
            rx, ry, rth = robot_pose

            # Control the robot to go to that pose
            ned_x, ned_y = self.pose2d_to_ned(rx, ry)
            print(ned_x, ned_y)

            self.client.moveToPositionAsync(ned_x, ned_y, self.height, 4).join()

            # rotate to yaw; go a little over to account for drift
            self.client.rotateToYawAsync(to_degrees(rth)+1.0).join()

            # After executing action, grab the current state set it to robot state
            ned_pose = self.ned_pose()
            next_rx, next_ry, next_rth = self.ned_to_pose2d(ned_pose)
            next_state.object_states[robot_id] = RobotState(robot_id,
                                                            (next_rx, next_ry, next_rth),
                                                            tuple(robot_state.objects_found),
                                                            action.name)
        reward = self.reward_model.sample(self.state, action, next_state, robot_id=robot_id)
        self.apply_transition(next_state)
        return reward
