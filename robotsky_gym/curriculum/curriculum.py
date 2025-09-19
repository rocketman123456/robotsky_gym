import os
import abc


class Curriculum:

    @abc.abstractmethod
    def update(self, env_ids, infos):
        raise NotImplementedError
