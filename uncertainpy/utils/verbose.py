class Verbose:
    def __init__(self, verbose_level="silent"):

        self.verbose_list = ["silent", "warnings", "info", "debug"]
        self.verbose_level = self.verbose_list.index(verbose_level)


    def __call__(self, message, verbose_level="silent"):
        if self.verbose_list.index(verbose_level) >= self.verbose_level:
            print message
