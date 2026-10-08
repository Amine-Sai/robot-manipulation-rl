import numpy as np

from robosuite.environments.manipulation.lift import Lift


class CustomLift(Lift):

    def reward(self, action=None):
        reward = 0.0

        if self._check_success():
            reward = 2.25

        elif self.reward_shaping:
            # Reward approaching the cube
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            gripper_site_pos = self.sim.data.site_xpos[
                self.robots[0].eef_site_id
            ]

            dist = np.linalg.norm(gripper_site_pos - cube_pos)
            reaching_reward = 1.0 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # Reward grasping
            grasped = self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.cube,
            )

            if grasped:
                reward += 0.25

                # Reward lifting after the cube is grasped
                cube_height = cube_pos[2]
                table_height = self.model.mujoco_arena.table_offset[2]

                lift_height = max(0.0, cube_height - table_height)

                lift_reward = np.clip(
                    lift_height / 0.04,
                    0.0,
                    1.5,
                )

                reward += lift_reward

        # Keep robosuite's original reward scaling
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        if self._check_success():
            reward += 100.0

        return reward