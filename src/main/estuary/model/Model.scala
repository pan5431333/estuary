package estuary.model

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.abs
import estuary.utils.PlotUtils
import org.apache.log4j.Logger

import scala.collection.mutable

/**
  * Created by mengpan on 2017/8/23.
  */
trait Model {
  val logger: Logger = Logger.getLogger("Model")

  protected var learningRate: Double = _
  protected var iterationTime: Int = _

  def setLearningRate(learningRate: Double): this.type = {
    this.learningRate = learningRate
    this
  }

  def setIterationTime(iterationTime: Int): this.type = {
    this.iterationTime = iterationTime
    this
  }

  def train(feature: DenseMatrix[Double], label: DenseVector[Double]): this.type

  def predict(feature: DenseMatrix[Double]): DenseVector[Double]

  def getLearningRate: Double = this.learningRate

  def getIterationTime: Int = this.iterationTime

  def getCostHistory: mutable.MutableList[Double]

  def plotCostHistory(): Unit = PlotUtils.plotCostHistory(getCostHistory)

}

object Model {
  def accuracy(label: DenseVector[Double], labelPredicted: DenseVector[Double]): Double = {
    val numCorrect = (0 until label.length)
      .map { index =>
        if (abs(label(index) - labelPredicted(index)) < scala.math.pow(10, -6)) 1 else 0
      }
      .count(_ == 1)
    numCorrect.toDouble / label.length.toDouble
  }
}
