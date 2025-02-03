"""
Driver class for Nintendo Switch Joy-Con
"""
import threading
import time
import numpy as np
from pynput.keyboard import Controller, Key, Listener
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

try:
    from pyjoycon import JoyCon as PyJoyCon
    from pyjoycon import get_R_id, get_L_id
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module pyjoycon, required to interface with Switch Joycon. "
    ) from exc

from pynput.keyboard import Controller, Key, Listener
import robosuite.macros as macros
from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix

class JoyCon(Device):
    def __init__(
        self,
        env,
        side="L",
        pos_sensitivity=4.0,
        rot_sensitivity=4.0
    ):
        super().__init__(env)

        print("Opening JoyCon device")
        if side == "R":
            joycon_id = get_R_id()
        elif side == "L":
            joycon_id = get_L_id()
        else:
            raise ValueError(f"Invalid side: {side}. Must be either 'R' or 'L'")
        
        self.device = PyJoyCon(*joycon_id)
        self.calibrate()

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # 6-DoF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False

        # launch a new listener thread to listen to JoyCon
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        super()._reset_internal_state()

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.single_click_and_hold = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.

        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dpos = self.control[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
            base_mode=int(self.base_mode),
        )

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """
        # TODO: display controls
        pass
        
    def run(self):
        """
        Listener method that keeps pulling new messages.
        """

        # t_last_click = -1

        while True:
            status = self.device.get_status()
            if status is not None and self._enabled:
                rotation = status['gyro']
                joystick = status['analog-sticks']['left']

                self.x = (joystick['horizontal'] - self.calibration_offset[6]) 
                self.y = (joystick['vertical'] - self.calibration_offset[7])
                self.z = 0
                self.roll = rotation['x']
                self.pitch = rotation['y']
                self.yaw = rotation['z']

                self._control = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]


    def calibrate(self):
        num_samples = 100
        samples = []
        for _ in range(num_samples):
            status = self.device.get_status()
            accel = status['accel']
            rot = status['gyro']
            joystick = status['analog-sticks']['left']

            samples.append([accel['x'], accel['y'], accel['z'], rot['x'], rot['y'], rot['z'], joystick['horizontal'], joystick['vertical']])
            time.sleep(0.01)
        
        self.calibration_offset = np.mean(samples, axis=0)

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse

        Returns:
            np.array: 6-DoF control value
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.

        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """
        pass

    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 50
        dpos = dpos * 125

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation
        
if __name__ == "__main__":
    env = None
    joycon = JoyCon(env, side="L")
    for i in range(100):
        print(joycon.control, joycon.control_gripper)
        time.sleep(0.02)
