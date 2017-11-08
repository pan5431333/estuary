package estuary.components.layers

import breeze.linalg.DenseMatrix
import estuary.components.initializer.WeightsInitializer
import estuary.components.regularizer.Regularizer

trait Layer extends Serializable{
  val numHiddenUnits: Int

  def copyStructure: Layer

  def forward(yPrevious: DenseMatrix[Double]): DenseMatrix[Double]

  def forwardForPrediction(yPrevious: DenseMatrix[Double]): DenseMatrix[Double]

  def backward(dYCurrent: DenseMatrix[Double], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double])

  def init(initializer: WeightsInitializer): DenseMatrix[Double]

  def getReguCost(regularizer: Option[Regularizer]): Double

  def setParam(param: DenseMatrix[Double]): Unit
}
