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

  def getCostHistory: mutable.MutableList[Double]

  def getLearningRate: Double = this.learningRate

  def getIterationTime: Int = this.iterationTime

  def plotCostHistory(): Unit = PlotUtils.plotCostHistory(getCostHistory)

}

object Model {
  def accuracy(label: DenseVector[Double], labelPredicted: DenseVector[Double]): Double = {
    val numCorrect = (0 until label.length)
      .map { index =>
        if (abs(label(index) - labelPredicted(index)) < math.pow(10, -6)) 1 else 0
      }
      .count(_ == 1)
    numCorrect.toDouble / label.length.toDouble
  }

  /**
    * Convert labels in a single vector to a matrix.
    * e.g. Vector(0, 1, 0, 1) => Matrix(Vector(1, 0, 1, 0), Vector(0, 1, 0, 1))
    * Vector(0, 1, 2) => Matrix(Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1))
    *
    * @param labelVector
    * @return
    */
  def convertVectorToMatrix(labelVector: DenseVector[Double]): DenseMatrix[Double] = {
    val labels = labelVector.toArray.toSet.toList.sorted //distinct elelents by toSet.

    val numLabels = labels.size
    val res = DenseMatrix.zeros[Double](labelVector.length, numLabels)

    for ((label, i) <- labels.zipWithIndex.par) {
      val helperVector = DenseVector.ones[Double](labelVector.length) * label
      res(::, i) := elementWiseEqualCompare(labelVector, helperVector)
    }
    res
  }

  def convertMatrixToVector(labelMatrix: DenseMatrix[Double], labelsMapping: List[Double]): DenseVector[Double] = {
    val labelsMappingVec = labelsMapping.toVector

    val res = DenseVector.zeros[Double](labelMatrix.rows)

    for (i <- 0 until labelMatrix.cols) {
      res :+= labelMatrix(::, i) * labelsMappingVec(i)
    }
    res
  }

  /**
    * Compare two vector for equality in element-wise.
    * e.g. a = Vector(1, 2, 3), b = Vector(1, 0, 0), then return Vector(1, 0, 0)
    *
    * @param a
    * @param b
    * @return
    */
  def elementWiseEqualCompare(a: DenseVector[Double], b: DenseVector[Double]): DenseVector[Double] = {
    assert(a.length == b.length, "a.length != b.length")
    val compareArr = a.toArray.zip(b.toArray).par.map { case (i, j) =>
      if (i == j) 1.0 else 0.0
    }.toArray
    DenseVector(compareArr)
  }
}
