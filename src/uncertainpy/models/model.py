class Model():
    """
simulation.run()
must return t and U
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

    def __call__(self, **parameters):
        """
        Run must return t, U
        """
        raise NotImplementedError("No __call__() method implemented")


    def run(self, **parameters):
        """
        Run must return t, U
        """
        raise NotImplementedError("No run() method implemented")
