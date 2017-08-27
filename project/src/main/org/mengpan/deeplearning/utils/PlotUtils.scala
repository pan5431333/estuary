package org.mengpan.deeplearning.utils

import javax.swing.JFrame

import org.jfree.chart.plot.PlotOrientation
import org.jfree.chart.{ChartFactory, ChartPanel, JFreeChart}
import org.jfree.data.xy.DefaultXYDataset

import scala.collection.mutable

/**
  * Created by mengpan on 2017/8/17.
  */
object PlotUtils {
  def plotCostHistory(costHistory: mutable.TreeMap[Int, Double]): Unit = {

    val x = costHistory.keys.toArray.map{_.toDouble}
    val y = costHistory.values.toArray[Double]

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
