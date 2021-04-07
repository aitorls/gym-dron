from gym.envs.registration import register

register(
    id='dron2dron_3DC-v0',
    entry_point='gym_dron.envs:Dron2dron_3DC',
)
register(
    id='dron2dron_2DC-v0',
    entry_point='gym_dron.envs:Dron2dron_2DC',
)
register(
    id='dron2dron_3DD-v0',
    entry_point='gym_dron.envs:Dron2dron_3DD',
)
register(
    id='dron2dron_2DD-v0',
    entry_point='gym_dron.envs:Dron2dron_2DD',
)
