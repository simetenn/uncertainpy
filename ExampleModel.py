from model import Model

# The class name and file name must be the same
class ExampleModel(Model):
    """
    The model must be able to handle these calls

    simulation = model()
    simulation.load()
    simulation.setParameters(parameters -> dictionary)
    simulation.run()
    simulation.save(current_process -> int)

    simulation.cmd()
    """
    def __init__(self):
        """
        Init must be able to be called with 0 arguments
        """
        Model.__init__(self)




    def load(self):
        """
        most likely not necesarry
        """
        pass


    def run(self):
        """
        run the model
        """


    def setParameters(self, parameters):
        """
        Parameters: dictionary with all parameters
        """
        # How the parameters are set


    def cmd(self):
        additional_cmds = []#additional_cmds needed to run this model from the command line
        return Model.cmd(self, additional_cmds)
