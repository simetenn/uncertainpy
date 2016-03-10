import numpy as np
import unittest
import chaospy as cp

from uncertainpy.evaluateNodeFunction import evaluateNodeFunction
from uncertainpy.features import TestingFeatures
from uncertainpy.models import TestingModel0d, TestingModel1d, TestingModel2d
from uncertainpy.parameters import Parameters


class TestEvaluateNodeFunction(unittest.TestCase):
    def setUp(self):
        parameterlist = [["a", 1, None],
                         ["b", 2, None]]

        self.parameters = Parameters(parameterlist)
        self.model = TestingModel1d(self.parameters)
        self.features = TestingFeatures()


        self.supress_model_output = True
        self.adaptive_model = False
        self.uncertain_parameters = ["a", "b"]
        self.node = [1, 2]
        self.feature_list = self.features.implementedFeatures()
        self.tmp_kwargs = {}


        # all_data = (cmds, supress_model_output, adaptive_model,
        #             node, tmp_parameter_names,
        #             feature_list, feature_cmd, kwargs)



    def setUpData(self):
        data = (self.model.cmd(),
                self.supress_model_output,
                self.adaptive_model,
                self.node,
                self.uncertain_parameters,
                self.feature_list,
                self.features.cmd(),
                self.tmp_kwargs)

        return data


    # {feature : (x, U, interpolation)}

    def test_evaluateNodeFunctionModel0d(self):
        self.model = TestingModel0d(self.parameters)

        result = evaluateNodeFunction(self.setUpData())


        self.assertEqual(set(self.feature_list + ["directComparison"]),
                         set(result.keys()))


        self.assertIsNone(result["directComparison"][0])
        self.assertEqual(result["directComparison"][1], 2)
        self.assertIsNone(result["directComparison"][2])


        self.assertIsNone(result["feature0d"][0])
        self.assertEqual(result["feature0d"][1], 1)
        self.assertIsNone(result["feature0d"][2])


        self.assertIsNone(result["feature1d"][0])
        self.assertTrue(np.array_equal(result["feature1d"][1],
                                       np.arange(0, 10)))
        self.assertIsNone(result["feature1d"][2])

        
        self.assertIsNone(result["feature2d"][0])
        self.assertTrue(np.array_equal(result["feature2d"][1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertIsNone(result["feature2d"][2])















if __name__ == "__main__":
    unittest.main()
