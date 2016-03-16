import numpy as np
import unittest
import chaospy as cp
import scipy.interpolate

from uncertainpy.evaluateNodeFunction import evaluateNodeFunction
from uncertainpy.features import TestingFeatures
from uncertainpy.models import TestingModel0d, TestingModel1d, TestingModel2d
from uncertainpy.models import TestingModel0dNoTime, TestingModel1dNoTime
from uncertainpy.models import TestingModel2dNoTime, TestingModelNoU
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
        self.node = [2, 3]
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


    def test_evaluateNodeFunctionModel0d(self):
        self.feature_list = []
        self.model = TestingModel0d(self.parameters)
        result = evaluateNodeFunction(self.setUpData())

        self.assertModel0d(result)


    def test_evaluateNodeFunctionModel1d(self):
        self.feature_list = []
        self.model = TestingModel1d(self.parameters)
        result = evaluateNodeFunction(self.setUpData())

        self.assertModel1d(result)


    def test_evaluateNodeFunctionModel2d(self):
        self.feature_list = []
        self.model = TestingModel2d(self.parameters)
        result = evaluateNodeFunction(self.setUpData())

        self.assertModel2d(result)


    def test_evaluateNodeFunctionModel0dAllFeatures(self):
        self.model = TestingModel0d(self.parameters)

        result = evaluateNodeFunction(self.setUpData())

        self.assertIn("feature0d", result.keys())
        self.assertEqual(len(result["feature0d"]), 3)

        self.assertEqual(result["feature0d"][0], 1)
        self.assertEqual(result["feature0d"][1], 1)
        self.assertIsNone(result["feature0d"][2])


        self.assertIn("feature1d", result.keys())
        self.assertEqual(len(result["feature1d"]), 3)

        self.assertEqual(result["feature1d"][0], 1)
        self.assertTrue(np.array_equal(result["feature1d"][1],
                                       np.arange(0, 10)))
        self.assertIsNone(result["feature1d"][2])


        self.assertIn("feature2d", result.keys())
        self.assertEqual(len(result["feature2d"]), 3)

        self.assertEqual(result["feature2d"][0], 1)
        self.assertTrue(np.array_equal(result["feature2d"][1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertIsNone(result["feature2d"][2])

        self.assertIn("featureInvalid", result.keys())
        self.assertEqual(len(result["featureInvalid"]), 3)
        self.assertEqual(result["feature0d"][0], 1)
        self.assertTrue(np.isnan(result["featureInvalid"][1]))
        self.assertIsNone(result["featureInvalid"][2])


    def test_evaluateNodeFunctionModel1dAllFeatures(self):
        self.model = TestingModel1d(self.parameters)

        result = evaluateNodeFunction(self.setUpData())

        self.assertModel1d(result)
        self.assertFeature0d(result)
        self.assertFeature1d(result)
        self.assertFeature2d(result)
        self.assertFeatureInvalid(result)


    def test_evaluateNodeFunctionModel2dAllFeatures(self):
        self.model = TestingModel2d(self.parameters)

        result = evaluateNodeFunction(self.setUpData())

        self.assertModel2d(result)
        self.assertFeature0d(result)
        self.assertFeature1d(result)
        self.assertFeature2d(result)
        self.assertFeatureInvalid(result)


    def test_evaluateNodeFunctionModel0dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = []

        self.model = TestingModel0d(self.parameters)
        with self.assertRaises(AttributeError):
            evaluateNodeFunction(self.setUpData())

    def test_evaluateNodeFunctionModel0dFeature0dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = ["feature0d"]

        self.model = TestingModel0d(self.parameters)
        with self.assertRaises(AttributeError):
            evaluateNodeFunction(self.setUpData())


    def test_evaluateNodeFunctionModel0dFeature1dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = ["feature1d"]

        self.model = TestingModel0d(self.parameters)
        with self.assertRaises(AttributeError):
            evaluateNodeFunction(self.setUpData())


    def test_evaluateNodeFunctionModel0dFeature2dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = ["feature2d"]

        self.model = TestingModel0d(self.parameters)
        with self.assertRaises(AttributeError):
            evaluateNodeFunction(self.setUpData())


    def test_evaluateNodeFunctionModel0dFeatureInvalidAdaptive(self):
        self.adaptive_model = True
        self.feature_list = ["featureInvalid"]

        self.model = TestingModel0d(self.parameters)
        with self.assertRaises(AttributeError):
            evaluateNodeFunction(self.setUpData())


    def test_evaluateNodeFunctionModel0dAllFeaturesAdaptive(self):
        self.adaptive_model = True

        self.model = TestingModel0d(self.parameters)
        with self.assertRaises(AttributeError):
            evaluateNodeFunction(self.setUpData())




    def test_evaluateNodeFunctionModel1dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = []

        self.model = TestingModel1d(self.parameters)
        result = evaluateNodeFunction(self.setUpData())

        self.assertIn("directComparison", result.keys())
        self.assertEqual(len(result["directComparison"]), 3)

        self.assertTrue(np.array_equal(result["directComparison"][0], np.arange(0, 10)))
        self.assertTrue(np.array_equal(result["directComparison"][1], np.arange(0, 10) + 5))
        self.assertIsNotNone(result["directComparison"][2])
        self.assertIsInstance(result["directComparison"][2],
                              scipy.interpolate.fitpack2.UnivariateSpline)



    def test_evaluateNodeFunctionModel1dFeature0dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = ["feature0d"]

        self.model = TestingModel1d(self.parameters)
        result = evaluateNodeFunction(self.setUpData())

        self.assertIn("directComparison", result.keys())
        self.assertEqual(len(result["directComparison"]), 3)

        self.assertTrue(np.array_equal(result["directComparison"][0], np.arange(0, 10)))
        self.assertTrue(np.array_equal(result["directComparison"][1], np.arange(0, 10) + 5))
        self.assertIsNotNone(result["directComparison"][2])
        self.assertIsInstance(result["directComparison"][2],
                              scipy.interpolate.fitpack2.UnivariateSpline)

        self.assertFeature0d


    def test_evaluateNodeFunctionModel1dFeature1dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = ["feature1d"]

        self.model = TestingModel1d(self.parameters)
        result = evaluateNodeFunction(self.setUpData())

        self.assertIn("directComparison", result.keys())
        self.assertEqual(len(result["directComparison"]), 3)

        self.assertTrue(np.array_equal(result["directComparison"][0], np.arange(0, 10)))
        self.assertTrue(np.array_equal(result["directComparison"][1], np.arange(0, 10) + 5))
        self.assertIsNotNone(result["directComparison"][2])
        self.assertIsInstance(result["directComparison"][2],
                              scipy.interpolate.fitpack2.UnivariateSpline)


        self.assertIn("feature1d", result.keys())
        self.assertEqual(len(result["feature1d"]), 3)
        self.assertTrue(np.array_equal(result["feature1d"][0],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(result["feature1d"][1], np.arange(0, 10)))
        self.assertIsInstance(result["feature1d"][2],
                              scipy.interpolate.fitpack2.UnivariateSpline)


    def test_evaluateNodeFunctionModel1dFeature2dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = ["feature2d"]

        self.model = TestingModel1d(self.parameters)
        with self.assertRaises(NotImplementedError):
            evaluateNodeFunction(self.setUpData())


    def test_evaluateNodeFunctionModelInvalidFeature1dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = ["featureInvalid"]

        self.model = TestingModel1d(self.parameters)
        result = evaluateNodeFunction(self.setUpData())

        self.assertIn("directComparison", result.keys())
        self.assertEqual(len(result["directComparison"]), 3)

        self.assertTrue(np.array_equal(result["directComparison"][0], np.arange(0, 10)))
        self.assertTrue(np.array_equal(result["directComparison"][1], np.arange(0, 10) + 5))
        self.assertIsNotNone(result["directComparison"][2])
        self.assertIsInstance(result["directComparison"][2],
                              scipy.interpolate.fitpack2.UnivariateSpline)

        self.assertFeatureInvalid(result)


    def test_evaluateNodeFunctionModel1dAllFeaturesAdaptive(self):
        self.adaptive_model = True

        self.model = TestingModel1d(self.parameters)
        with self.assertRaises(NotImplementedError):
            evaluateNodeFunction(self.setUpData())




    def test_evaluateNodeFunctionModel2dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = []

        self.model = TestingModel2d(self.parameters)
        with self.assertRaises(NotImplementedError):
            evaluateNodeFunction(self.setUpData())


    def test_evaluateNodeFunctionModel2dFeature0dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = ["feature0d"]

        self.model = TestingModel2d(self.parameters)
        with self.assertRaises(NotImplementedError):
            evaluateNodeFunction(self.setUpData())


    def test_evaluateNodeFunctionModel2dFeature1dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = ["feature1d"]

        self.model = TestingModel2d(self.parameters)
        with self.assertRaises(NotImplementedError):
            evaluateNodeFunction(self.setUpData())


    def test_evaluateNodeFunctionModel2dFeature2dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = ["feature2d"]

        self.model = TestingModel2d(self.parameters)
        with self.assertRaises(NotImplementedError):
            evaluateNodeFunction(self.setUpData())


    def test_evaluateNodeFunctionModel2dFeature2dAdaptive(self):
        self.adaptive_model = True
        self.feature_list = ["featureInvalid"]

        self.model = TestingModel2d(self.parameters)
        with self.assertRaises(NotImplementedError):
            evaluateNodeFunction(self.setUpData())



    def test_evaluateNodeFunctionModel2dAllFeaturesAdaptive(self):
        self.adaptive_model = True

        self.model = TestingModel2d(self.parameters)
        with self.assertRaises(NotImplementedError):
            evaluateNodeFunction(self.setUpData())





    def test_evaluateNodeFunctionModel0dNoTime(self):
        self.feature_list = []
        self.model = TestingModel0dNoTime(self.parameters)
        result = evaluateNodeFunction(self.setUpData())


        self.assertIn("directComparison", result.keys())
        self.assertEqual(len(result["directComparison"]), 3)

        self.assertIsNone(result["directComparison"][0])
        self.assertEqual(result["directComparison"][1], 3)
        self.assertIsNone(result["directComparison"][2])


    def test_evaluateNodeFunctionModel1dNoTime(self):
        self.feature_list = []
        self.model = TestingModel1dNoTime(self.parameters)
        result = evaluateNodeFunction(self.setUpData())


        self.assertIn("directComparison", result.keys())
        self.assertEqual(len(result["directComparison"]), 3)

        self.assertIsNone(result["directComparison"][0])
        self.assertTrue(np.array_equal(result["directComparison"][1], np.arange(0, 10) + 5))
        self.assertIsNone(result["directComparison"][2])


    def test_evaluateNodeFunctionModel2dNoTime(self):
        self.feature_list = []
        self.model = TestingModel2dNoTime(self.parameters)
        result = evaluateNodeFunction(self.setUpData())


        self.assertIn("directComparison", result.keys())
        self.assertEqual(len(result["directComparison"]), 3)

        self.assertIsNone(result["directComparison"][0])
        self.assertTrue(np.array_equal(result["directComparison"][1],
                                       np.array([np.arange(0, 10) + 2,
                                                 np.arange(0, 10) + 3])))
        self.assertIsNone(result["directComparison"][2])






    def test_evaluateNodeFunctionModel0dNoTimeAdaptive(self):
        self.feature_list = []
        self.adaptive_model = True
        self.model = TestingModel0dNoTime(self.parameters)

        with self.assertRaises(AttributeError):
            evaluateNodeFunction(self.setUpData())


    def test_evaluateNodeFunctionModel1dNoTimeAdaptive(self):
        self.feature_list = []
        self.adaptive_model = True
        self.model = TestingModel1dNoTime(self.parameters)

        with self.assertRaises(AttributeError):
            evaluateNodeFunction(self.setUpData())


    def test_evaluateNodeFunctionModel2dNoTimeAdaptive(self):
        self.feature_list = []
        self.adaptive_model = True
        self.model = TestingModel2dNoTime(self.parameters)

        with self.assertRaises(NotImplementedError):
            evaluateNodeFunction(self.setUpData())


    def test_evaluateNodeFunctionModel1dNoTimefeatures1dAdaptive(self):
        self.feature_list = ["feature1d"]
        self.adaptive_model = True
        self.model = TestingModel1dNoTime(self.parameters)

        with self.assertRaises(AttributeError):
            evaluateNodeFunction(self.setUpData())



    def test_evaluateNodeFunctionModelNoU(self):
        self.feature_list = []
        self.model = TestingModelNoU(self.parameters)

        with self.assertRaises(RuntimeError):
            evaluateNodeFunction(self.setUpData())






    def assertModel0d(self, result):
        self.assertIn("directComparison", result.keys())
        self.assertEqual(len(result["directComparison"]), 3)

        self.assertEqual(result["directComparison"][0], 1)
        self.assertEqual(result["directComparison"][1], 3)
        self.assertIsNone(result["directComparison"][2])




    def assertModel1d(self, result):
        self.assertIn("directComparison", result.keys())
        self.assertEqual(len(result["directComparison"]), 3)

        self.assertTrue(np.array_equal(result["directComparison"][0], np.arange(0, 10)))
        self.assertTrue(np.array_equal(result["directComparison"][1], np.arange(0, 10) + 5))
        self.assertIsNone(result["directComparison"][2])


    def assertModel2d(self, result):
        self.assertIn("directComparison", result.keys())
        self.assertEqual(len(result["directComparison"]), 3)

        self.assertTrue(np.array_equal(result["directComparison"][0], np.arange(0, 10)))
        self.assertTrue(np.array_equal(result["directComparison"][1],
                                       np.array([np.arange(0, 10) + 2,
                                                 np.arange(0, 10) + 3])))
        self.assertIsNone(result["directComparison"][2])


    def assertFeature0d(self, result):
        self.assertIn("feature0d", result.keys())
        self.assertEqual(len(result["feature0d"]), 3)
        self.assertTrue(np.array_equal(result["feature0d"][0],
                                       np.arange(0, 10)))
        self.assertEqual(result["feature0d"][1], 1)
        self.assertIsNone(result["feature0d"][2])


    def assertFeature1d(self, result):
        self.assertIn("feature1d", result.keys())
        self.assertEqual(len(result["feature1d"]), 3)

        self.assertTrue(np.array_equal(result["feature1d"][0],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(result["feature1d"][1],
                                       np.arange(0, 10)))
        self.assertIsNone(result["feature1d"][2])


    def assertFeature2d(self, result):
        self.assertIn("feature2d", result.keys())
        self.assertEqual(len(result["feature2d"]), 3)

        self.assertTrue(np.array_equal(result["feature2d"][0],
                                       np.arange(0, 10)))
        self.assertTrue(np.array_equal(result["feature2d"][1],
                                       np.array([np.arange(0, 10),
                                                 np.arange(0, 10)])))
        self.assertIsNone(result["feature2d"][2])


    def assertFeatureInvalid(self, result):
        self.assertIn("featureInvalid", result.keys())
        self.assertEqual(len(result["featureInvalid"]), 3)
        self.assertIsNone(result["featureInvalid"][0])
        self.assertTrue(np.isnan(result["featureInvalid"][1]))
        self.assertIsNone(result["featureInvalid"][2])


if __name__ == "__main__":
    unittest.main()
