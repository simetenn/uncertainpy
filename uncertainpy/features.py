from uncertainpy.spikes import Spikes

class Features:
    def __init__(self, spikes):

        if not isinstance(spikes, Spikes):
            raise TypeError("spikes must be a Spikes object")

        self.spikes = spikes
        # self.implemented_features = ["nrSpikes", "timeBeforeFirstSpike"]

        self.utility_methods = ["calculateFeature", "calculateFeatures",
                                "calculateAllFeatures", "__init__", "implementedFeatures"]


    def nrSpikes(self):
        return self.spikes.nr_spikes


    def timeBeforeFirstSpike(self):
        return self.spikes.spikes[0].t_max


    def calculateFeature(self, feature_name):
        if not callable(getattr(self, feature_name)):
            raise NotImplementedError("%s is not a implemented feature" % (feature_name))

        results = {feature_name: getattr(self, feature_name)()}
        for feature in self.features_to_examine:
            results[feature] = getattr(self, feature)()

        return results


    def calculateFeatures(self, feature_names):
        results = []
        for feature_name in self.implementedFeatures():
            results.append(self.calculateFeature(feature_name))

        return results


    def calculateAllFeatures(self):
        results = {}
        for feature in self.implementedFeatures():
            results[feature] = getattr(self, feature)()

        return results


    def implementedFeatures(self):
        """
        Return a list of all callable methods in feature
        """
        return [method for method in dir(self) if callable(getattr(self, method)) and method not in self.utility_methods]
