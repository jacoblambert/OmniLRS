"""Load only the requested environment, without importing unused terrain engines."""

from importlib import import_module


class ROS2_LabManagerFactory:
    def __init__(self):
        self._lab_managers = {}

    def register(self, name, lab_manager):
        # Preserve explicit class registration; module/name tuples defer imports.
        self._lab_managers[name] = lab_manager

    def __call__(self, cfg, **kwargs):
        manager = self._lab_managers[cfg["environment"]["name"]]
        if isinstance(manager, tuple):
            module, name = manager
            manager = getattr(import_module(module), name)
        return manager(environment_cfg=cfg["environment"], **kwargs)


ROS2_LMF = ROS2_LabManagerFactory()
ROS2_LMF.register("Lunalab", ("src.environments_wrappers.ros2.lunalab_ros2", "ROS_LunalabManager"))
ROS2_LMF.register("Lunaryard", ("src.environments_wrappers.ros2.lunaryard_ros2", "ROS_LunaryardManager"))
ROS2_LMF.register("LargeScale", ("src.environments_wrappers.ros2.largescale_ros2", "ROS_LargeScaleManager"))
