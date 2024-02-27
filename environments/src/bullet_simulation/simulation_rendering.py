import pdb


class SimulationRendering:

    def __init__(self, bullet_client, display):

        self._display = None  # flag that triggers rendering

        self._init_attributes(bullet_client=bullet_client, display=display)

    @property
    def display(self):
        return self._display

    @display.setter
    def display(self, val):
        self._display = val

    def _init_attributes(self, bullet_client, display):

        self._display = display
        self._init_camera(bullet_client=bullet_client)

    def _init_camera(self, bullet_client):
        camera_distance = 1.  # Distance de la caméra par rapport à l'objet
        camera_pitch = 30.0  # Angle d'inclinaison de la caméra (en degrés)
        camera_yaw = -30.0  # Angle de lacet de la caméra (en degrés)
        camera_target_position = [0, 0, 0]  # Position que la caméra cible

        bullet_client.resetDebugVisualizerCamera(camera_distance, camera_pitch, camera_yaw, camera_target_position)


