# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import matplotlib
matplotlib.use("TkAgg")
import rclpy
from policy_translation.srv import NetworkPT
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from utils.voice import Voice
import sys
import select
import numpy as np
import cv2
import sensor_msgs.msg
import math
import pickle
import glob
import json
import csv
import os.path
import os
import matplotlib.pyplot as plt
from PIL import Image

# Default robot position. You don't need to change this
DEFAULT_UR5_JOINTS  = [105.0, -30.0, 120.0, 90.0, 60.0, 90.0]
# Evaluate headless or not
HEADLESS            = False
# This is a debug variable... 
USE_SHAPE_SIZE      = True
# Run on the test data, or start the simulator in manual mode 
# (manual mode will allow you to generate environments and type in your own commands)
RUN_ON_TEST_DATA    = True
# How many of the 100 test-data do you want to test?
NUM_TESTED_DATA     = 100
# Where to find the normailization?
NORM_PATH           = "../GDrive/normalization_v2.pkl"
# Where to find the VRep scene file. This has to be an absolute path. 
VREP_SCENE          = "../GDrive/NeurIPS2020.ttt"
VREP_SCENE          = os.getcwd() + "/" + VREP_SCENE

class Simulator(object):
    def __init__(self, args=None):
        rclpy.init(args=args)
        self.node       = rclpy.create_node("VRepSimTest")
        self.srv_prx_nn = self.node.create_client(NetworkPT, "/network")

        self.node.get_logger().info("Service not available, waiting...")
        while not self.srv_prx_nn.wait_for_service(timeout_sec=1.0):
            pass
        self.node.get_logger().info("... service found!")

        self.pyrep = PyRep()
        self.pyrep.launch(VREP_SCENE, headless=HEADLESS)
        self.camera = VisionSensor("kinect_rgb_full")
        self.pyrep.start()

        self.trajectory    = None
        self.global_step   = 0
        self.normalization = pickle.load(open(NORM_PATH, mode="rb"), encoding="latin1")
        self.voice         = Voice(load=False)  

        self.shape_size_replacement = {}
        self.shape_size_replacement["58z29D2omoZ_2.json"] = "spill everything into the large curved dish"
        self.shape_size_replacement["P1VOZ4zk4NW_2.json"] = "fill a lot into the small square basin"
        self.shape_size_replacement["KOVJZ4Npy4G_2.json"] = "fill a small amount into the big round pot"
        self.shape_size_replacement["wjqQmB74rnr_2.json"] = "pour all of it into the large square basin"
        self.shape_size_replacement["LgVK8qXGowA_2.json"] = "fill a little into the big round bowl"
        self.shape_size_replacement["JZ90qm46ooP_2.json"] = "fill everything into the biggest rectangular bowl"
    
    def loadNlpCSV(self, path):
        self.nlp_dict = {}
        with open(path, "r") as fh:
            csvreader = csv.reader(fh, delimiter=",")
            for line in csvreader:
                if line[1] != "":
                    self.nlp_dict[line[0]+"_1.json"] = line[1]
                    self.nlp_dict[line[0]+"_2.json"] = line[2]

    def shutdown(self):
        self.pyrep.stop()
        self.pyrep.shutdown()

    def _getCameraImage(self):
        rgb_obs = self.camera.capture_rgb()
        rgb_obs = (np.asarray(rgb_obs) * 255).astype(dtype=np.uint8)
        rgb_obs = np.flip(rgb_obs, (2))
        return rgb_obs
    
    def _getSimulatorState(self):
        _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="getState@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        return s
    
    def _stopRobotMovement(self):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="stopRobotMovement@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
    
    def _getRobotState(self):
        _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="getState@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        state = np.take(np.asarray(s), indices=[0,1,2,3,4,5,30], axis=0)  
        return state.tolist()

    def _setRobotJoints(self, joints):
            result = self.pyrep.script_call(function_name_at_script_name="setRobotJoints@control_script",
                                            script_handle_or_type=1,
                                            ints=(), floats=joints, strings=(), bytes="")

    def _setJointVelocityFromTarget(self, joints):
        _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="setJointVelocityFromTarget@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=joints, strings=(), bytes="")

    def _setJointVelocityFromTarget_Direct(self, joints):
        _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="setJointVelocityFromTarget_Direct@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=joints, strings=(), bytes="")
    
    def _dropBall(self, b_id):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="dropBall@control_script",
                                        script_handle_or_type=1,
                                        ints=(b_id,), floats=(), strings=(), bytes="")
    
    def _evalPouring(self):
        i, _, _, _ = self.pyrep.script_call(function_name_at_script_name="evalPouring@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        return i
    
    def _graspedObject(self):
        i, _, _, _ = self.pyrep.script_call(function_name_at_script_name="graspedObject@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        if i[0] >= 0:
            return True
        return False
    
    def _setRobotInitial(self, joints):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="setRobotJoints@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=joints, strings=(), bytes="")
    
    def _graspClosestContainer(self):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="graspClosestContainer@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")

    def _randomizeLight(self):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="randomizeLight@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")

    def _resetEnvironment(self):
        self.pyrep.stop()
        self.pyrep.start()
    
    def _createEnvironment(self, ints, floats):
        result = self.pyrep.script_call(
            function_name_at_script_name="generateScene@control_script",
            script_handle_or_type=1,
            ints=ints, 
            floats=floats, 
            strings=(), 
            bytes=""
        )

    def _getClosesObject(self):
        oid, dist, _, _ = self.pyrep.script_call(function_name_at_script_name="getClosesObject@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        return oid, dist

    def dtype_with_channels_to_cvtype2(self, dtype, n_channels):
        numpy_type_to_cvtype = {'uint8': '8U', 'int8': '8S', 'uint16': '16U',
                                     'int16': '16S', 'int32': '32S', 'float32': '32F',
                                     'float64': '64F'}
        numpy_type_to_cvtype.update(dict((v, k) for (k, v) in numpy_type_to_cvtype.items()))
        return '%sC%d' % (numpy_type_to_cvtype[dtype.name], n_channels)

    def cv2_to_imgmsg(self, cvim, encoding="passthrough"):
        if not isinstance(cvim, (np.ndarray, np.generic)):
            raise TypeError('Your input type is not a numpy array')
        img_msg = sensor_msgs.msg.Image()
        img_msg.height = cvim.shape[0]
        img_msg.width = cvim.shape[1]
        if len(cvim.shape) < 3:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, 1)
        else:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, cvim.shape[2])
        if encoding == 'passthrough':
            img_msg.encoding = cv_type
        else:
            img_msg.encoding = encoding
            # Verify that the supplied encoding is compatible with the type of the OpenCV image
            if self.cvtype_to_name[self.encoding_to_cvtype2(encoding)] != cv_type:
                raise CvBridgeError('encoding specified as %s, but image has incompatible type %s'
                                    % (encoding, cv_type))
        if cvim.dtype.byteorder == '>':
            img_msg.is_bigendian = True
        img_msg.data.frombytes(cvim.tobytes())
        img_msg.step = len(img_msg.data) // img_msg.height

        return img_msg
    
    def predictTrajectory(self, voice, state, cnt):
        norm         = np.take(self.normalization["values"], indices=[0,1,2,3,4,5,30], axis=1)
        image        = self._getCameraImage()

        robot_state    = state
        robot_state[6] = self.last_gripper

        req          = NetworkPT.Request()
        req.image    = self.cv2_to_imgmsg(image[:,:,::-1])
        req.language = self._generalizeVoice(voice)
        req.robot    = self.normalize([robot_state], norm[0,:], norm[1,:]).flatten().tolist()
        req.reset    = cnt == 1
        req.plot     = False

        future       = self.srv_prx_nn.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        try:
            result = future.result()
        except Exception as e:
            self.node.get_logger().info('Service call failed %r' % (e,))
            return False
        
        trajectory = np.asarray(result.trajectory).reshape(-1, 7)
        trajectory = self.restoreValues(trajectory, norm[0,:], norm[1,:])
        phase      = float(result.phase)
        return trajectory, phase
    
    def normalize(self, value, v_min, v_max):
        if type(value) == list:
            value = np.asarray(value)
        if (value.shape[1] != v_min.shape[0] or v_min.shape[0] != v_max.shape[0] or 
            len(value.shape) != 2 or len(v_min.shape) != 1 or len(v_max.shape) != 1):
            raise ArrayDimensionMismatch()
        value = np.copy(value)
        v_min = np.tile(np.expand_dims(v_min, 0), [value.shape[0], 1])
        v_max = np.tile(np.expand_dims(v_max, 0), [value.shape[0], 1])
        value = (value - v_min) / (v_max - v_min)
        return value
    
    def restoreValues(self, value, v_min, v_max):
        if (value.shape[1] != v_min.shape[0] or v_min.shape[0] != v_max.shape[0] or 
            len(value.shape) != 2 or len(v_min.shape) != 1 or len(v_max.shape) != 1):
            print("Array dimensions are not matching!")

        value = np.copy(value)
        v_min = np.tile(np.expand_dims(v_min, 0), [value.shape[0], 1])
        v_max = np.tile(np.expand_dims(v_max, 0), [value.shape[0], 1])
        value = value * (v_max - v_min) + v_min
        return value

    def _generalizeVoice(self, voice):
        return voice

    def _mapObjectIDs(self, oid):
        if oid == 154:
            return 1
        elif oid == 155:
            return 2
        elif oid == 156:
            return 3
        elif oid == 113:
            return 1
        elif oid == 118:
            return 2
        elif oid == 124:
            return 3
        elif oid == 130:
            return 4
        elif oid == 136:
            return 5
        elif oid == 115:
            return 6
        elif oid == 119:
            return 7
        elif oid == 125:
            return 8
        elif oid == 131:
            return 9
        elif oid == 137:
            return 10
        elif oid == 148:
            return 11
        elif oid == 147:
            return 12
        elif oid == 146:
            return 13
        elif oid == 145:
            return 14
        elif oid == 143:
            return 15
        elif oid == 152:
            return 16
        elif oid == 151:
            return 17
        elif oid == 150:
            return 18
        elif oid == 149:
            return 19
        elif oid == 144:
            return 20

    def _getTargetPosition(self, data):
        state  = self._getSimulatorState()
        tcp    = state[12:14]
        target = data["target/id"]
        tp     = data["target/type"]
        if tp == "cup":
            cups   = data["ints"][2+data["ints"][0]:]
            t_id   = [i for i in range(data["ints"][1]) if cups[i] == target][0] + data["ints"][0]
            t_pos  = data["floats"][t_id*3:t_id*3+2]
        else:
            bowls  = data["ints"][2:2+data["ints"][0]]
            t_id   = [i for i in range(data["ints"][0]) if bowls[i] == target][0]
            t_pos  = data["floats"][t_id*3:t_id*3+2]

        dist   = np.sqrt( np.power(tcp[0] - t_pos[0], 2) + np.power(tcp[1] - t_pos[1], 2) )

        closest       = list(self._getClosesObject())
        closest[0][0] = self._mapObjectIDs(closest[0][0])
        closest[0][1] = self._mapObjectIDs(closest[0][1])
        result = {}
        result["target"]     = t_pos
        result["tid"]        = target
        result["tid/actual"] = closest
        result["current"]    = tcp
        result["distance"]   = dist
        return result

    def _maybeDropBall(self, state):
        res = 0
        if state[5] > 3.0:
            self._dropBall(1)
            res = 1
        if state[5] > 3.0 and self.last_rotation > state[5]:
            self._dropBall(2)
            res = 2
        self.last_rotation = state[5]
        return res

    def _getLanguateInformation(self, voice, phs):
        def _quantity(voice):
            res = 0
            for word in self.voice.synonyms["little"]:
                if voice.find(word) >= 0:
                    res = 1
            for word in self.voice.synonyms["much"]:
                if voice.find(word) >= 0:
                    res = 2
            return res

        def _difficulty(voice):
            if phs == 2:
                voice  = " ".join(voice.split()[4:])
            shapes = self.voice.synonyms["round"] + self.voice.synonyms["square"]
            colors = self.voice.synonyms["small"] + self.voice.synonyms["large"]  
            sizes  = self.voice.synonyms["red"]   + self.voice.synonyms["green"] + self.voice.synonyms["blue"] + self.voice.synonyms["yellow"] + self.voice.synonyms["pink"]

            shapes_used = 0
            for word in shapes:
                if voice.find(word) >= 0:
                    shapes_used = 1
            colors_used = 0
            for word in colors:
                if voice.find(word) >= 0:
                    colors_used = 1
            sizes_used = 0
            for word in sizes:
                if voice.find(word) >= 0:
                    sizes_used = 1
            return shapes_used + colors_used + sizes_used

        data = {}
        data["original"] = voice
        data["features"] = _difficulty(voice)
        data["quantity"] = _quantity(voice)
        return data

    def valPhase1(self, files, feedback=True):
        successfull = 0
        val_data    = {}
        nn_trajectory  = []
        ro_trajectory  = []
        for fid, fn in enumerate(files):
            print("Phase 1 Run {}/{}".format(fid, len(files)))
            eval_data = {}
            with open(fn + "1.json", "r") as fh:
                data = json.load(fh)            

            gt_trajectory = np.asarray(data["trajectory"])
            self._resetEnvironment()
            self._createEnvironment(data["ints"], data["floats"])
            self._setRobotInitial(gt_trajectory[0,:])
            self.pyrep.step()

            eval_data["language"] = self._getLanguateInformation(data["voice"], 1)
            eval_data["trajectory"] = {"gt": [], "state": []}
            eval_data["trajectory"]["gt"] = gt_trajectory.tolist()

            cnt   = 0
            phase = 0.0
            self.last_gripper = 0.0
            th = 1.0
            while phase < th and cnt < int(gt_trajectory.shape[0] * 1.5):
                state = self._getRobotState() if feedback else gt_trajectory[-1 if cnt >= gt_trajectory.shape[0] else cnt,:]
                cnt += 1
                tf_trajectory, phase = self.predictTrajectory(data["voice"], state, cnt)
                r_state    = tf_trajectory[-1,:]
                eval_data["trajectory"]["state"].append(r_state.tolist())
                r_state[6] = r_state[6] 
                nn_trajectory.append(r_state)
                ro_trajectory.append(self._getRobotState())
                self.last_gripper = r_state[6]
                self._setJointVelocityFromTarget(r_state)
                self.pyrep.step()
                if r_state[6] > 0.5 and "locations" not in eval_data.keys():
                    eval_data["locations"] = self._getTargetPosition(data)

            eval_data["success"] = False
            if self._graspedObject():
                eval_data["success"] = True
                successfull += 1
            val_data[data["name"]] = eval_data
            
        return successfull, val_data
    
    def valPhase2(self, files, feedback=True):
        successfull         = 0
        val_data            = {}
        for fid, fn in enumerate(files):
            print("Phase 2 Run {}/{}".format(fid, len(files)))
            eval_data = {}
            fpath     = fn + "2.json"
            filename  = os.path.basename(fpath)
            with open(fpath, "r") as fh:
                data = json.load(fh)
            gt_trajectory = np.asarray(data["trajectory"])
            
            if USE_SHAPE_SIZE and filename in self.shape_size_replacement.keys():
                data["voice"] = self.shape_size_replacement[filename]

            self._resetEnvironment()
            self._createEnvironment(data["ints"], data["floats"])
            self._setRobotInitial(gt_trajectory[0,:])
            self.pyrep.step()
            self._graspClosestContainer()
            self.pyrep.step()

            self.last_gripper  = 1.0
            self.last_rotation = 0.0

            eval_data["language"] = self._getLanguateInformation(data["voice"], 2)
            eval_data["trajectory"] = {"gt": [], "state": []}
            eval_data["trajectory"]["gt"] = gt_trajectory.tolist()

            cnt   = 0
            phase = 0.0
            th = 1.0
            while phase < th and cnt < int(gt_trajectory.shape[0] * 1.5):
                state = self._getRobotState() if feedback else gt_trajectory[-1 if cnt >= gt_trajectory.shape[0] else cnt,:]
                cnt += 1
                tf_trajectory, phase = self.predictTrajectory(data["voice"], self._getRobotState(), cnt)
                r_state              = tf_trajectory[-1,:]
                eval_data["trajectory"]["state"].append(r_state.tolist())
                r_state[6]           = r_state[6]
                self._setJointVelocityFromTarget(r_state)
                self.last_gripper = r_state[6]
                dropped = self._maybeDropBall(r_state)
                if dropped == 1 and "locations" not in eval_data.keys():
                    eval_data["locations"] = self._getTargetPosition(data)
                self.pyrep.step()

            presult                 = self._evalPouring()
            eval_percentage         = np.sum(presult) / float(len(presult))
            eval_data["ball_array"] = presult
            if eval_percentage > 0.5:
                successfull += 1
                eval_data["success"] = True
            else:
                eval_data["success"] = False
            val_data[data["name"]] = eval_data
            
        return successfull, val_data

    def evalDirect(self, runs):
        files = glob.glob("../GDrive/testdata/*_1.json")
        self.node.get_logger().info("Using data directory with {} files".format(len(files)))
        files = files[:runs]
        files = [f[:-6] for f in files]
        self.node.get_logger().info("Running validation on {} files".format(len(files)))

        data = {}
        s_p1, e_data        = self.valPhase1(files)
        data["phase_1"]     = e_data
        s_p2, e_data        = self.valPhase2(files)
        data["phase_2"]     = e_data

        self.node.get_logger().info("Testing Picking: {}/{} ({:.1f}%)".format(s_p1,  runs, 100.0 * float(s_p1)/float(runs)))
        self.node.get_logger().info("Testing Pouring: {}/{} ({:.1f}%)".format(s_p2,  runs, 100.0 * float(s_p2)/float(runs)))

        p1_names = data["phase_1"].keys()
        p2_names = data["phase_2"].keys()
        names = [n for n in p1_names if n in p2_names]
        c_p2  = 0
        for n in names:
            if data["phase_1"][n]["success"] and data["phase_2"][n]["success"]:
                c_p2  += 1

        self.node.get_logger().info("Whole Task: {}/{} ({:.1f}%)".format(c_p2,  len(names), 100.0 * float(c_p2)  / float(len(names))))

        with open("val_result.json", "w") as fh:
            json.dump(data, fh)
    
    def _generateEnvironment(self):
        def genPosition(prev):
            px = 0
            py = 0
            done = False        
            while not done:
                done = True        
                px = np.random.uniform(-0.9, 0.35)
                py = np.random.uniform(-0.9, 0.35)
                dist = np.sqrt(px**2 + py**2)
                if dist < 0.5 or dist > 0.9:
                    done = False
                for o in prev:
                    if np.sqrt((px - o[0])**2 + (py - o[1])**2) < 0.25:
                        done = False
                if px > 0 and py > 0:
                    done = False
                angle = -45
                r_px  = px * np.cos(np.deg2rad(angle)) + py * np.sin(np.deg2rad(angle))
                r_py  = py * np.cos(np.deg2rad(angle)) - px * np.sin(np.deg2rad(angle))
                if r_py > 0.075:
                    done = False
            return [px, py]
        self._setRobotJoints(np.deg2rad(DEFAULT_UR5_JOINTS))

        ncups  = np.random.randint(1,3)
        nbowls = np.random.randint(ncups,5)
        bowls  = np.random.choice(20, size=nbowls, replace=False) + 1
        cups   = np.random.choice(3, size=ncups, replace=False) + 1
        ints   = [nbowls, ncups] + bowls.tolist() + cups.tolist()
        floats = []

        prev   = []
        for i in range(nbowls + ncups):
            prev.append(genPosition(prev))
            floats += prev[-1]
            if i < nbowls and bowls[i] > 10:
                floats += [np.random.uniform(-math.pi/4.0,  math.pi/4.0)]
            else:
                floats += [0.0]

        self._createEnvironment(ints, floats)
        self.node.get_logger().info("Created new environment")
        return ints, floats

    def simplifyVoice(self, voice):
        simple = []
        for word in voice.split(" "):
            if word in self.voice.basewords.keys():
                simple.append(self.voice.basewords[word])
        return " ".join(simple)

    def parseInput(self, d_in):
        if d_in == "q":
            return False
        if d_in == "g":
            self.rm_voice     = ""
            self.last_gripper = 0.0
            self._generateEnvironment()
        if d_in == "r":
            self.rm_voice     = ""
            self.last_gripper = 0.0
            self.node.get_logger().info("Resetting robot")
            self._resetEnvironment()
        elif d_in.startswith("t "):
            self.rm_voice = d_in[2:]
            self.cnt      = 0
            print("Running Task: " + self.rm_voice)
        elif self.rm_voice != "" and  d_in == "":
            self.cnt += 1
            tf_trajectory, phase = self.predictTrajectory(self.rm_voice, self._getRobotState(), self.cnt)
            r_state              = tf_trajectory[-1,:]
            self.last_gripper    = r_state[6]
            self._setJointVelocityFromTarget(r_state)
            self._maybeDropBall(r_state)

            if phase >=0.95:
                self.node.get_logger().info("Finished running trajectory with " + str(self.cnt) + " steps")
                self._stopRobotMovement()
                self.rm_voice = ""

        return True
    
    def runManually(self):
        self.rm_voice = ""
        run  = True
        while run:
            self.pyrep.step()
            if select.select([sys.stdin,],[],[],0.0)[0]:
                line = sys.stdin.readline()
                run = self.parseInput(line[:-1])
            else:
                self.parseInput("")    
        print("Shutting down...")
  
if __name__ == "__main__":
    sim = Simulator()
    if RUN_ON_TEST_DATA:
        sim.evalDirect(runs=NUM_TESTED_DATA)
    else:
        sim.runManually()
    sim.shutdown()
    