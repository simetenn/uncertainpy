from uncertainpy.spikes import Spikes

class Features:
    def __init__(self, spikes=None):

        if not isinstance(spikes, Spikes) and None:
            raise TypeError("spikes must be None or a Spikes object")

        self.spikes = spikes
        # self.implemented_features = ["nrSpikes", "timeBeforeFirstSpike"]

        self.utility_methods = ["calculateFeature",
                                "calculateFeatures",
                                "calculateAllFeatures",
                                "__init__",
                                "implementedFeatures",
                                "setSpikes"]


    def setSpikes(self, spikes):
        if not isinstance(spikes, Spikes) and None:
            raise TypeError("spikes must be None or a Spikes object")

        self.spikes = spikes

    def nrSpikes(self):
        return self.spikes.nr_spikes


    def timeBeforeFirstSpike(self):
        return self.spikes.spikes[0].t_max

    def averageAPOvershoot(self):
        sum_AP_overshoot = 0
        for spike in self.spikes:
            sum_AP_overshoot += spike.U_max
        return sum_AP_overshoot/float(self.spikes.nr_spikes)


    def calculateFeature(self, feature_name):
        if not callable(getattr(self, feature_name)):
            raise NotImplementedError("%s is not a implemented feature" % (feature_name))

        return getattr(self, feature_name)()


    def calculateFeatures(self, feature_names):
        results = {}
        for feature in feature_names:
            results[feature] = self.calculateFeature(feature)

        return results


    def calculateAllFeatures(self):
        results = {}
        for feature in self.implementedFeatures():
            results[feature] = self.calculateFeature(feature)

        return results


    def implementedFeatures(self):
        """
        Return a list of all callable methods in feature
        """
        return [method for method in dir(self) if callable(getattr(self, method)) and method not in self.utility_methods]
