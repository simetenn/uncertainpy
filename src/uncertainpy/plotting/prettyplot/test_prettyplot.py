import numpy as np
import matplotlib.pyplot as plt
import unittest


from prettyplot import prettyPlot, prettyBar

# TODO compare all plots to an existing plot

class TestPrettyPlot(unittest.TestCase):
    def setUp(self):
        self.time= np.arange(0, 10)
        self.values = np.arange(0, 10) + 1


    def test_prettyPlotX(self):

        prettyPlot(self.U)

        plt.close()


    def test_prettyPlotXY(self):
        prettyPlot(self.time, self.U)
        plt.close()


    def test_prettyPlotXYColor(self):
        prettyPlot(self.time, self.U)
        plt.close()


    def test_prettyPlotFalseNewFigure(self):
        prettyPlot(self.U)
        prettyPlot(self.time, self.U, new_figure=False)
        plt.close()


    def test_prettyPlotFalseNewFigure2(self):
        prettyPlot(self.time, self.U, new_figure=False)
        plt.close()


class TestPrettyBar(unittest.TestCase):
    def setUp(self):
        self.values = np.arange(2, 7)
        self.error = np.arange(2, 7)*0.1
        self.labels = ["1", "2", "3", "4", "5"]


    def test_PrettyBar(self):
        prettyBar(self.U)
        plt.close()


    def test_PrettyBarError(self):
        prettyBar(self.U, self.error)
        plt.close()


    def test_PrettyBarLabels(self):
        prettyBar(self.U, xlabels=self.labels)
        plt.close()


    def test_prettyBarFalseNewFigure(self):
        prettyBar(self.U)
        prettyBar(self.U, new_figure=False)
        plt.close()


    def test_prettyPlotFalseNewFigure2(self):
        prettyBar(self.U, new_figure=False)
        plt.close()


if __name__ == "__main__":
    unittest.main()
