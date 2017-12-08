package estuary.helper

import javax.swing.JFrame

import org.jfree.chart.plot.PlotOrientation
import org.jfree.chart.{ChartFactory, ChartPanel, JFreeChart}
import org.jfree.data.xy.DefaultXYDataset

/**
  * Created by mengpan on 2017/8/17.
  */
object PlotUtils {
  def plotCostHistory(costHistory: Seq[Double]): Unit = {

    val x = costHistory.indices.map(_.toDouble).toArray
    val y = costHistory.toArray[Double]

    val data = Array(x, y)

    val xyDataset: DefaultXYDataset = new DefaultXYDataset()
    xyDataset.addSeries("Iteration v.s. Cost", data)

    val jFreeChart: JFreeChart = ChartFactory.createScatterPlot("Cost History",
      "Iteration", "Cost", xyDataset, PlotOrientation.VERTICAL, true, false, false
    )


    val panel = new ChartPanel(jFreeChart, true)

    val frame = new JFrame()

    frame.add(panel)
    frame.setBounds(50, 50, 800, 600)
    frame.setVisible(true)
  }
}
