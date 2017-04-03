class Model(object):
    """
run(**parameters) must return t and U
    """
    def __init__(self,
                 adaptive_model=False,
                 xlabel="",
                 ylabel=""):
        """
        """

        self.adaptive_model = adaptive_model
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.name = self.__class__.__name__


    @property
    def run(self):
        return self._run


    @run.setter
    def run(self, new_run):
        self._run = new_run
        self.name = new_run.__name__


    def set_parameters(self, **parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])



    def __call__(self, **parameters):
        """
        Run must return t, U
        """
        raise NotImplementedError("No __call__() method implemented")


    def _run(self, **parameters):
        """
        Run must return t, U
        """
        raise NotImplementedError("No run() method implemented")
