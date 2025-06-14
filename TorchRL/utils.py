from __future__ import annotations
import logging
from typing import Collection, Hashable
import numpy as np
from typing import Dict, Iterable, List, Tuple
logger = logging.getLogger(__name__)
from soulsgym.games.game import StaticGameData
import torch
import gymnasium as gym

class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Create a flattened observation space based on an example obs
        self.transformer = GameStateTransformer()
        obs_sample = self.observation(env.reset()[0])  # first obs from env
        flat_dim = obs_sample.shape[0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
        self.current_phase = 1

    def observation(self, obs):
        transform = True
        self.current_phase = obs['phase']
        if transform:
            observation = self.transformer.transform(obs)
            observation = torch.tensor(observation, dtype=torch.float32)
        else:
            flat = []
            for v in obs.values():
                arr = torch.tensor(v, dtype=torch.float32).flatten()
                flat.append(arr)
            observation = torch.cat(flat, dim=0)
        #print(observation)
        return observation

class OneHotEncoder:
    """Encode categorical data as one hot numpy arrays.

    Just like the sklearn encoder (which this class imitates), the encoder first has to be fit to
    data before it can be used to convert between the representations.
    """

    def __init__(self, allow_unknown: bool = False):
        """Initialize the lookup dictionaries.

        Args:
            allow_unknown: Flag to allow unknown categories.
        """
        self._key_to_index_dict: dict[Hashable, int] = dict()
        self._index_to_key_dict: dict[int, Hashable] = dict()
        self.dim: int = 0
        self.allow_unknown = allow_unknown

    def fit(self, data: Collection[Hashable]):
        """Fit the encoder to the training data.

        Args:
            data: A collection of hashable categories
        """
        for idx, key in enumerate(data):
            self._key_to_index_dict[key] = idx
            self._index_to_key_dict[idx] = key
        self.dim = len(data)

    def __call__(self, data: Hashable) -> np.ndarray:
        """Alias for :meth:`.OneHotEncoder.transform`.

        Args:
            data: A categorical data sample.

        Returns:
            The corresponding one hot encoded array.
        """
        return self.transform(data)

    def transform(self, data: Hashable) -> np.ndarray:
        """Transform categorical data to a one hot encoded array.

        Args:
            data: A categorical data sample.

        Returns:
            The corresponding one hot encoded array.

        Raises:
            ValueError: An unknown category was provided without setting allow_unknown to `True`.
        """
        if data not in self._key_to_index_dict.keys():
            if not self.allow_unknown:
                raise ValueError("OneHotEncoder received an unknown category.")
            logger.warning(f"Unknown key {data} encountered")
            return np.zeros(self.dim, dtype=np.float32)
        x = np.zeros(self.dim, dtype=np.float32)
        x[self._key_to_index_dict[data]] = 1
        return x

    def inverse_transform(self, data: np.ndarray) -> Hashable:
        """Transform one-hot encoded data to its corresponding category.

        Args:
            data: A one-hot encoded data sample.

        Returns:
            The corresponding category.

        Raises:
            ValueError: An unknown category was provided.
        """
        key = np.where(data == 1)
        if not len(key[0]) == 1 or len(data) != self.dim:
            raise ValueError("OneHotEncoder received an unknown category.")
        return self._index_to_key_dict[key[0][0]]


class GameStateTransformer:
    """Transform ``SoulsGym`` observations into a numerical representation.

    The transformer allows the consistent binning of animations and encodes pose data into a more
    suitable representation.
    """

    SOULSGYM_STEP_TIME = 0.1
    space_coords_low = np.array([110.0, 540.0, -73.0])
    space_coords_high = np.array([190.0, 640.0, -55.0])
    space_coords_diff = space_coords_high - space_coords_low

    def __init__(self, game_id: str = "DarkSoulsIII", boss_id: str = "iudex"):
        """Initialize the one-hot encoders and set up attributes for the stateful transformation.

        Args:
            game_id: Game ID.
            boss_id: Boss ID.
        """
        self.game_id = game_id
        self.boss_id = boss_id
        self.game_data = StaticGameData(game_id)
        # Initialize player one-hot encoder
        self.player_animation_encoder = OneHotEncoder(allow_unknown=True)
        p_animations = [a["ID"] for a in self.game_data.player_animations.values()]
        filtered_player_animations = unique(map(self.filter_player_animation, p_animations))
        self.player_animation_encoder.fit(filtered_player_animations)
        # Initialize boss one-hot encoder
        self.boss_animation_encoder = OneHotEncoder(allow_unknown=True)
        iudex_animations = self.game_data.boss_animations[self.boss_id]["all"]
        boss_animations = [a["ID"] for a in iudex_animations.values()]
        filtered_boss_animations = unique(
            map(lambda x: self.filter_boss_animation(x)[0], boss_animations)
        )
        self.boss_animation_encoder.fit(filtered_boss_animations)
        # Initialize stateful attributes
        self._current_time = 0.0
        self._acuumulated_time = 0.0
        self._last_animation = None

    def transform(self, obs: Dict) -> np.ndarray:
        """Transform a game observation with a stateful conversion.

        Warning:
            This function is assumed to be called with successive ``SoulsGym`` observations. If the
            next observation is not part of the same trajectory, :meth:`.GameStateTransformer.reset`
            has to be called.

        Args:
            obs: The input observation.

        Returns:
            A transformed observation as a numerical array.
        """
        # The final observation has the following entries:
        # 0-3: player_hp, player_sp, boss_hp, boss_distance
        # 4-15: player_pos, player_rot, boss_pos, boss_rot, camera_rot
        # 16-17: player_animation_duration, boss_animation_duration
        # 17-48: player_animation_onehot
        # 49-73: boss_animation_onehot
        obs = self._unpack_obs(obs)
        player_animation = self.filter_player_animation(obs["player_animation"])
        player_animation_onehot = self.player_animation_encoder(player_animation)
        boss_animation_onehot, boss_animation_duration = self.boss_animation_transform(obs)
        animation_times = [obs["player_animation_duration"], boss_animation_duration]
        return np.concatenate(
            (
                self._common_transforms(obs),
                animation_times,
                player_animation_onehot,
                boss_animation_onehot,
            ),
            dtype=np.float32,
        )

    def stateless_transform(self, obs: Dict) -> np.ndarray:
        """Transform a game observation with a stateless conversion.

        Boss and player animations are filtered and binned, but not accumulated correctly.

        Args:
            obs: The input observation.

        Returns:
            A transformed observation as a numerical array.
        """
        obs = self._unpack_obs(obs)
        animation_times = np.concatenate(
            (obs["player_animation_duration"], obs["boss_animation_duration"])
        )
        player_animation = self.filter_player_animation(obs["player_animation"])
        player_animation_onehot = self.player_animation_encoder(player_animation)
        boss_animation = self.filter_boss_animation(obs["boss_animation"])[0]
        boss_animation_onehot = self.boss_animation_encoder(boss_animation)
        return np.concatenate(
            (
                self._common_transforms(obs),
                animation_times,
                player_animation_onehot,
                boss_animation_onehot,
            ),
            dtype=np.float32,
        )

    def _common_transforms(self, obs: Dict) -> np.ndarray:
        player_hp = obs["player_hp"] / obs["player_max_hp"]
        player_sp = obs["player_sp"] / obs["player_max_sp"]
        boss_hp = obs["boss_hp"] / obs["boss_max_hp"]
        player_pos = (obs["player_pose"][:3] - self.space_coords_low) / self.space_coords_diff
        player_rot = rad2vec(obs["player_pose"][3])
        boss_pos = (obs["boss_pose"][:3] - self.space_coords_low) / self.space_coords_diff
        boss_rot = rad2vec(obs["boss_pose"][3])
        camera_angle = np.arctan2(obs["camera_pose"][3], obs["camera_pose"][4])
        camera_rot = rad2vec(camera_angle)
        # 50 is a normalization guess
        boss_distance = np.linalg.norm(obs["boss_pose"][:2] - obs["player_pose"][:2]) / 50
        args = (
            [player_hp, player_sp, boss_hp, boss_distance],
            player_pos,
            player_rot,
            boss_pos,
            boss_rot,
            camera_rot,
        )
        return np.concatenate(args, dtype=np.float32)

    def reset(self):
        """Reset the stateful attributed of the transformer.

        See :meth:`.GameStateTransformer.boss_animation_transform`.
        """
        self._current_time = 0.0
        self._acuumulated_time = 0.0
        self._last_animation = None

    def boss_animation_transform(self, obs: Dict) -> Tuple[np.ndarray, float]:
        """Transform the observation's boss animation into a one-hot encoding and a duration.

        Since we are binning the boss animations, we have to sum the durations of binned animations.
        This requires the transformer to be stateful to keep track of previous animations.

        Note:
            To correctly transform animations after an episode has finished, users have to call
            :meth:`.GameStateTransformer.reset` in between.

        Args:
            obs: The input observation.

        Returns:
            A tuple of the current animation as one-hot encoding and the animation duration.
        """
        boss_animation, is_filtered = self.filter_boss_animation(obs["boss_animation"])
        if not is_filtered:
            self._acuumulated_time = 0.0
            self._current_time = 0.0
            self._last_animation = boss_animation
            return self.boss_animation_encoder(boss_animation), obs["boss_animation_duration"]
        if obs["boss_animation"] != self._last_animation:
            self._last_animation = obs["boss_animation"]
            # The animation has changed. obs["boss_animation_duration"] now contains the duration
            # of the new animation. We have to calculate the final duration of the previous
            # animation by adding the time from the step at t-1 until the animation first changed to
            # the accumulated time.
            remaining_duration = self.SOULSGYM_STEP_TIME - obs["boss_animation_duration"]
            self._acuumulated_time = self._current_time + remaining_duration
        boss_animation_time = obs["boss_animation_duration"] + self._acuumulated_time
        self._current_time = boss_animation_time
        return self.boss_animation_encoder(boss_animation), boss_animation_time

    @staticmethod
    def filter_player_animation(animation: int) -> int:
        """Bin common player animations.

        Player animations that essentially constitute the same state are binned into a single
        category to reduce the state space. The new labels are in the range of 1xx to avoid
        collisions with other animation labels.

        Args:
            animation: Player animation ID.

        Returns:
            The binned player animation.
        """
        if animation in [0, 1, 2, 3, 4]:  # <Add-x> animations
            return 100
        if animation in [17, 18, 19, 23, 24, 25, 26]:  # <Idle, Move, None, Run-x>
            return 101
        if animation in [27, 28]:  # <Quick-x>
            return 102
        if animation in [39, 40]:  # <LandLow, Land>
            return 103
        if animation in [41, 42]:  # <LandFaceDown, LandFaceUp>
            return 104
        if animation in [43, 44, 45, 46, 47, 48, 49]:  # <Fall-x>
            return 105
        return animation

    def filter_boss_animation(self, animation: int) -> Tuple[int, bool]:
        """Bin boss movement animations into a single animation.

        Boss animations that essentially constitute the same state are binned into a single category
        to reduce the state space. The new labels are in the range of 1xx to avoid collisions with
        other animation labels.

        Args:
            animation: Boss animation ID.

        Returns:
            The animation name and a flag set to True if it was binned (else False).
        """
        # <WalkFront, WalkLeft, WalkRight, WalkBack, TurnRight90, TurnRight180, TurnLeft90,
        # TurnLeft180>
        if animation in [19, 20, 21, 22, 24, 25, 26, 27]:
            return 100, True
        return animation, False

    @staticmethod
    def _unpack_obs(obs: Dict) -> Dict:
        """Unpack numpy arrays of float observations.

        Args:
            obs: The initial observation.

        Returns:
            The observation with unpacked floats.
        """
        scalars = [
            "player_hp",
            "player_sp",
            "boss_hp",
            "boss_animation_duration",
            "player_animation_duration",
        ]
        for key in scalars:
            if isinstance(obs[key], np.ndarray):
                obs[key] = obs[key].item()
        arrays = ["player_pose", "boss_pose", "camera_pose"]
        for key in arrays:
            if not isinstance(obs[key], np.ndarray):
                assert isinstance(obs[key], list)
                obs[key] = np.array(obs[key])
        return obs


def unique(seq: Iterable) -> List:
    """Create a list of unique elements from an iterable.

    Args:
        seq: Iterable sequence.

    Returns:
        The list of unique items.
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def wrap2pi(x: float) -> float:
    """Project an angle in radians to the interval of [-pi, pi].

    Args:
        x: The angle in radians.

    Returns:
        The angle restricted to the interval of [-pi, pi].
    """
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def rad2vec(x: float) -> np.ndarray:
    """Convert an angle in radians to a [sin, cos] vector.

    Args:
        x: The angle in radians.

    Returns:
        The encoded orientation as [sin, cos] vector.
    """
    return np.array([np.sin(x), np.cos(x)])