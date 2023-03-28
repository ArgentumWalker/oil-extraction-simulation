from src.utils.map_generation.generate_linked_graph import generate_graph
from src.render.render import RenderSettings
from src.envs.envs import BlackOilEnv
import numpy as np


if __name__ == "__main__":
    settings = RenderSettings()
    settings.display_ground = True
    settings.display_wells = True

    env = BlackOilEnv()
    state = env.reset()
    env.render(settings)

    for _ in range(3):
        action = np.random.randint(0, state.shape[1]), np.random.randint(0, state.shape[0])
        state, reward, done = env.step(action)
        env.render(settings)