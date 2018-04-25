import numpy as np
import unittest

from uncertainpy.utils import lengths, none_to_nan, contains_nan
from uncertainpy.utils import is_regular, set_nan


class TestLengths(unittest.TestCase):
    def test_nested(self):
        evaluation = [[1, 2, 3], [4, [1, [5], 5]], 6, 1]
        result = lengths(evaluation)

        self.assertEqual(result, [4, 3, 2, 3, 1])

    def test_nested_2(self):
        evaluation = [[1, 2, 3], [4, [1, [5], 5]], 6, [1]]
        result = lengths(evaluation)

        self.assertEqual(result, [4, 3, 2, 3, 1, 1])


    def test_simple(self):
        evaluation = np.array([1, 2, 3])
        result = lengths(evaluation)

        self.assertEqual(result, [3])


    def test_simple_nested(self):
        evaluation = np.array([[1, 2, 3]])
        result = lengths(evaluation)

        self.assertEqual(result, [1, 3])


    def test_empty(self):
        evaluation = [np.nan, [4, [1, [5], 5]], 6, []]
        result = lengths(evaluation)

        self.assertEqual(result, [4, 2, 3, 1, 0])



class TestSetNan(unittest.TestCase):
    def test_list(self):
        values = [1, 1, 1, 1, 1]
        set_nan(values, 1)
        self.assertEqual(values, [1, np.nan, 1, 1, 1])

    def test_nested(self):
        values = [[], [1, 2, 3], 2, [[1, 2], 3, 3]]
        set_nan(values, [1, 2])
        set_nan(values, [3, 0, 1])
        self.assertEqual(values, [[], [1, 2, np.nan], 2, [[1, np.nan], 3, 3]])


class TestNoneToNan(unittest.TestCase):
    def test_none_to_nan_empty_list(self):
        values_irregular = np.array([[], np.array([1, 2, 3]), None, [[], None, 3]])

        result = none_to_nan(values_irregular)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], [])
        self.assertTrue(np.isnan(result[2]))
        self.assertTrue(np.array_equal(result[1], np.array([1, 2, 3])))
        self.assertEqual(result[3][0], [])
        self.assertTrue(np.isnan(result[3][1]))
        self.assertEqual(result[3][2], 3)


    def test_none_to_nan_int(self):
        result = none_to_nan(2)

        self.assertEqual(result, 2)


    def test_none_to_nan_str(self):
        result = none_to_nan("time")

        self.assertEqual(result, "time")


    def test_none_to_nan_list(self):
        result = none_to_nan([1, 2, 3])

        self.assertEqual(result, [1, 2, 3])



    def test_none_to_nan_long(self):
        result = none_to_nan([np.arange(1, 100), np.arange(1, 100), None, np.arange(1, 100).tolist()])

        self.assertTrue(np.array_equal(result[0], np.arange(1, 100)))
        self.assertTrue(np.array_equal(result[1], np.arange(1, 100)))
        self.assertTrue(np.isnan(result[2]))
        self.assertEqual(result[3], np.arange(1, 100).tolist())


    def test_none_to_nan_nested(self):
        values_irregular = [None, [1, 2, 3], None]

        result = none_to_nan(values_irregular)

        self.assertEqual(len(result), 3)
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], [1, 2, 3])
        self.assertTrue(np.isnan(result[2]))


    def test_none_to_nan_nested_none(self):
        values_irregular = np.array([None, np.array([None, np.array([1, 2, 3])])])

        result = none_to_nan(values_irregular)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[1]), 2)
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1][0]))
        self.assertTrue(np.array_equal(result[1][1], [1, 2, 3]))


    def test_none_to_nan_nested_no_none(self):
        values_irregular = np.array([np.array([1, 2, 3]), np.array([1, 2, 3]),
                                     np.array([1, 2, 3]), np.array([1, 2, 3])])

        values_correct = np.array([np.array([1, 2, 3]), np.array([1, 2, 3]),
                                   np.array([1, 2, 3]), np.array([1, 2, 3])])

        result = none_to_nan(values_irregular)

        self.assertTrue(np.array_equal(result, values_correct))


    def test_none_to_nan_nested_array_single(self):
        values_irregular = np.array([None, np.array([np.array(1), np.array(2), np.array(3)]),
                                     None, np.array([np.array(1), np.array(2), np.array(3)])])

        result = none_to_nan(values_irregular)

        self.assertEqual(len(result), 4)
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[2]))

        self.assertEqual(len(result[1]), 3)
        self.assertTrue(np.array_equal(result[1],  np.array([np.array(1), np.array(2), np.array(3)])))
        self.assertEqual(len(result[3]), 3)
        self.assertTrue(np.array_equal(result[3],  np.array([np.array(1), np.array(2), np.array(3)])))



    def test_none_to_nan_nested_array_single_no_none(self):
        values_irregular = np.array([np.array(1), np.array(2), np.array(3)])

        result = none_to_nan(values_irregular)

        values_correct = np.array([np.array(1), np.array(2), np.array(3)])

        self.assertTrue(np.array_equal(result, np.array([np.array(1), np.array(2), np.array(3)])))


        values_irregular = np.array([None, None, None])

        result = none_to_nan(values_irregular)

        self.assertEqual(len(result), 3)
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))




class TestContainsNoneOrNan(unittest.TestCase):
    def test_simple(self):
        values = [1, 2, 3]
        result = contains_nan(values)
        self.assertFalse(result)


    def test_simple_nan(self):
        values = [1, np.nan, 3]
        result = contains_nan(values)
        self.assertTrue(result)


    def test_array_nan(self):
        values = np.array([np.array(1), np.array(np.nan), np.array(3)])
        result = contains_nan(values)
        self.assertTrue(result)


    def test_numpy_float_nan(self):
        values = [np.float64(1), np.float64(np.nan), np.float64(3)]
        result = contains_nan(values)
        self.assertTrue(result)

    def test_nested(self):
        values = [[1, 2], 3]
        result = contains_nan(values)
        self.assertFalse(result)


    def test_nested_nan(self):
        values = [[1, np.nan], 3]
        result = contains_nan(values)
        self.assertTrue(result)


    def test_deeply_nested(self):
        values = [[1, [1, 4, 2]], 3, 3, []]
        result = contains_nan(values)
        self.assertFalse(result)


    def test_deeply_nested_nan(self):
        values = [[1, [1, 4, 2]], np.nan, 3, []]
        result = contains_nan(values)
        self.assertTrue(result)

    def test_simple_none(self):
        values = [1, None, 3]
        result = contains_nan(values)
        self.assertTrue(result)

    def test_nested_none(self):
        values = [[1, None], 3]
        result = contains_nan(values)
        self.assertTrue(result)


    def test_deeply_nested_none(self):
        values = [[1, [1, 4, 2]], None, 3, []]
        result = contains_nan(values)
        self.assertTrue(result)


    def test_int(self):
        values = 1
        result = contains_nan(values)
        self.assertFalse(result)

    def test_int_nan(self):
        values = np.nan
        result = contains_nan(values)
        self.assertTrue(result)


    def test_int_none(self):
        values = None
        result = contains_nan(values)
        self.assertTrue(result)


class TestIsRegular(unittest.TestCase):
    def test_regular(self):
        values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = is_regular(values)
        self.assertTrue(result)


    def test_regular_nested(self):
        values = [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]
        result = is_regular(values)
        self.assertTrue(result)


    def test_regular_nan(self):
        values = [np.nan, [4, 5, 6], [7, 8, 9]]
        result = is_regular(values)
        self.assertFalse(result)


    def test_regular_one_list(self):
        values = [np.nan, None, [7, 8, 9]]
        result = is_regular(values)
        self.assertFalse(result)


    def test_regular_all_nan(self):
        values = [np.nan, None, np.nan]
        result = is_regular(values)
        self.assertTrue(result)


    def test_irregular(self):
        values = [1, 2, [1]]
        result = is_regular(values)
        self.assertFalse(result)


    def test_irregular_2(self):
        values = [[1, 2, 3], [1, 2]]
        result = is_regular(values)
        self.assertFalse(result)


    def test_irregular_nan(self):
        values = [[1, 2, 3], [1, np.nan], None]
        result = is_regular(values)
        self.assertFalse(result)


    def test_irregular_nested(self):
        values = [[[1, 2], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]

        result = is_regular(values)
        self.assertFalse(result)


    def test_int(self):
        values = 1

        result = is_regular(values)
        self.assertTrue(result)


    def test_nan(self):
        values = np.nan

        result = is_regular(values)
        self.assertTrue(result)






# Commented out since function currently does not work
# and is not needed
# class TestOnlyNoneOrNan(unittest.TestCase):
#     def test_array_nan(self):
#         values = np.array([np.array(1), np.array(np.nan), np.array(3)])
#         result = only_none_or_nan(values)
#         self.assertFalse(result)


#     def test_numpy_float_nan(self):
#         values = [np.float64(1), np.float64(np.nan), np.float64(3)]
#         result = only_none_or_nan(values)
#         self.assertFalse(result)


#     def test_simple(self):
#         values = [1, 2, 3]
#         result = only_none_or_nan(values)
#         self.assertFalse(result)


#     def test_one_nan(self):
#         values = [1, np.nan, 3]
#         result = only_none_or_nan(values)
#         self.assertFalse(result)


#     def test_simple_nan(self):
#         values = [np.nan, np.nan, None]
#         result = only_none_or_nan(values)
#         self.assertTrue(result)


#     def test_empty(self):
#         values = [1, 2, 3, [], [[1], 2, 3]]
#         result = only_none_or_nan(values)
#         self.assertFalse(result)


#     def test_simple_nan_empty(self):
#         values = [np.nan, np.nan, None, [], [[None], np.nan, np.nan]]
#         result = only_none_or_nan(values)
#         self.assertFalse(result)


#     def test_int(self):
#         values = 1
#         result = only_none_or_nan(values)
#         self.assertFalse(result)


#     def test_none(self):
#         values = None
#         result = only_none_or_nan(values)
#         self.assertTrue(result)







# Old none_to_nan_regularize tests

    # def test_none_to_nan_regularize_empty_list(self):
    #     values_irregular = np.array([[], np.array([1, 2, 3]), None, [[], None, 3]])

    #     result = none_to_nan_regularize(values_irregular)

    #     print result
    #     self.assertEqual(len(result), 4)
    #     self.assertEqual(result[0], [])
    #     self.assertTrue(np.isnan(result[2]))
    #     self.assertTrue(np.array_equal(result[1], np.array([1, 2, 3])))
    #     self.assertEqual(result[3][0], [])
    #     self.assertTrue(np.isnan(result[3][1]))
    #     self.assertEqual(result[3][2], 3)


    # def test_none_to_nan_regularize_int(self):

    #     result = none_to_nan_regularize(2)

    #     self.assertEqual(result, 2)


    # def test_none_to_nan_regularize_list(self):

    #     result = none_to_nan_regularize([1, 2, 3])

    #     self.assertEqual(result, [1, 2, 3])


    # def test_none_to_nan_regularize_nested(self):
    #     values_irregular = [None, [1, 2, 3], None]

    #     result = none_to_nan_regularize(values_irregular)

    #     print "==================================="
    #     print result

    #     self.assertEqual(len(result), 3)
    #     self.assertTrue(np.isnan(result[0]))
    #     self.assertEqual(result[1], [1, 2, 3])
    #     self.assertTrue(np.isnan(result[2]))


    # def test_none_to_nan_regularize_nested_none(self):
    #     values_irregular = np.array([None, np.array([None, np.array([1, 2, 3])])])

    #     result = none_to_nan_regularize(values_irregular)

    #     self.assertEqual(len(result), 2)
    #     self.assertEqual(len(result[1]), 2)
    #     self.assertTrue(np.isnan(result[0]))
    #     self.assertTrue(np.isnan(result[1][0]))
    #     self.assertTrue(np.array_equal(result[1][1], [1, 2, 3]))


    # def test_none_to_nan_regularize_nested_no_none(self):
    #     values_irregular = np.array([np.array([1, 2, 3]), np.array([1, 2, 3]),
    #                                  np.array([1, 2, 3]), np.array([1, 2, 3])])

    #     values_correct = np.array([np.array([1, 2, 3]), np.array([1, 2, 3]),
    #                                np.array([1, 2, 3]), np.array([1, 2, 3])])

    #     result = none_to_nan_regularize(values_irregular)

    #     self.assertTrue(np.array_equal(result, values_correct))


    # def test_none_to_nan_regularize_nested_array_single(self):
    #     values_irregular = np.array([None, np.array([np.array(1), np.array(2), np.array(3)]),
    #                                  None, np.array([np.array(1), np.array(2), np.array(3)])])

    #     result = none_to_nan_regularize(values_irregular)

    #     self.assertEqual(len(result), 4)
    #     self.assertTrue(np.isnan(result[0]))
    #     self.assertTrue(np.isnan(result[2]))

    #     self.assertEqual(len(result[1]), 3)
    #     self.assertTrue(np.array_equal(result[1],  np.array([np.array(1), np.array(2), np.array(3)])))
    #     self.assertEqual(len(result[3]), 3)
    #     self.assertTrue(np.array_equal(result[3],  np.array([np.array(1), np.array(2), np.array(3)])))



    # def test_none_to_nan_regularize_nested_array_single_no_none(self):
    #     values_irregular = np.array([np.array(1), np.array(2), np.array(3)])

    #     result = none_to_nan_regularize(values_irregular)

    #     values_correct = np.array([np.array(1), np.array(2), np.array(3)])

    #     self.assertTrue(np.array_equal(result, np.array([np.array(1), np.array(2), np.array(3)])))


    #     values_irregular = np.array([None, None, None])

    #     result = none_to_nan_regularize(values_irregular)

    #     self.assertEqual(len(result), 3)
    #     self.assertTrue(np.isnan(result[0]))
    #     self.assertTrue(np.isnan(result[1]))
    #     self.assertTrue(np.isnan(result[2]))

            # def test_none_to_nan(self):

    #     values_irregular = np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])])

    #     result = none_to_nan(values_irregular)

    #     values_correct = np.array([[np.nan, np.nan, np.nan], [1, 2, 3],
    #                           [np.nan, np.nan, np.nan], [1, 2, 3]])


    #     result = np.array(result)
    #     print result
    #     self.assertTrue(((result == values_correct) | (np.isnan(result) & np.isnan(values_correct))).all())



    #     values_irregular = np.array([None,
    #                             np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])]),
    #                             np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])]),
    #                             np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])]),
    #                             None])

    #     result = none_to_nan(values_irregular)

    #     values_correct = np.array([[[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
    #                            [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    #                           [[np.nan, np.nan, np.nan], [1, 2, 3],
    #                            [np.nan, np.nan, np.nan], [1, 2, 3]],
    #                           [[np.nan, np.nan, np.nan], [1, 2, 3],
    #                            [np.nan, np.nan, np.nan], [1, 2, 3]],
    #                           [[np.nan, np.nan, np.nan], [1, 2, 3],
    #                            [np.nan, np.nan, np.nan], [1, 2, 3]],
    #                           [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
    #                            [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]])

    #     result = np.array(result)
    #     self.assertTrue(((result == values_correct) | (np.isnan(result) & np.isnan(values_correct))).all())



    #     values_irregular = np.array([np.array([1, 2, 3]), np.array([1, 2, 3]),
    #                             np.array([1, 2, 3]), np.array([1, 2, 3])])

    #     result = none_to_nan(values_irregular)

    #     result = np.array(result)
    #     self.assertTrue(np.array_equal(result, values_irregular))



    #     values_irregular = np.array([None, np.array([np.array(1), np.array(2), np.array(3)]),
    #                             None, np.array([np.array(1), np.array(2), np.array(3)])])

    #     result = none_to_nan(values_irregular)

    #     values_correct = np.array([[np.nan, np.nan, np.nan], [1, 2, 3],
    #                           [np.nan, np.nan, np.nan], [1, 2, 3]])

    #     result = np.array(result)
    #     self.assertTrue(((result == values_correct) | (np.isnan(result) & np.isnan(values_correct))).all())



    #     values_irregular = np.array([np.array(1), np.array(2), np.array(3)])

    #     result = none_to_nan(values_irregular)

    #     values_correct = np.array([np.array(1), np.array(2), np.array(3)])

    #     result = np.array(result)
    #     self.assertTrue(np.array_equal(result, values_irregular))


    #     values_irregular = np.array([None, None, None])

    #     result = none_to_nan(values_irregular)

    #     values_correct = np.array([np.nan, np.nan, np.nan])

    #     result = np.array(result)

    #     self.assertTrue(np.all(np.isnan(result)))
    #     self.assertEqual(len(result), 3)




    # def test_none_to_nan_int(self):

    #     result = none_to_nan(2)

    #     self.assertEqual(result, 2)


    # def test_none_to_nan_list(self):

    #     result = none_to_nan([1, 2, 3])

    #     self.assertEqual(result, [1, 2, 3])


    # def test_none_to_nan_nested(self):
    #     values_irregular = [None, [1, 2, 3], None]

    #     result = none_to_nan(values_irregular)

    #     self.assertEqual(len(result), 3)
    #     self.assertTrue(np.isnan(result[0]))
    #     self.assertEqual(result[1], [1, 2, 3])
    #     self.assertTrue(np.isnan(result[2]))


    # def test_none_to_nan_nested_none(self):
    #     values_irregular = np.array([None, np.array([None, np.array([1, 2, 3])])])

    #     result = none_to_nan(values_irregular)

    #     self.assertEqual(len(result), 2)
    #     self.assertEqual(len(result[1]), 2)
    #     self.assertTrue(np.isnan(result[0]))
    #     self.assertTrue(np.isnan(result[1][0]))
    #     self.assertTrue(np.array_equal(result[1][1], [1, 2, 3]))
