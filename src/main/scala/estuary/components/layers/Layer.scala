package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import estuary.components.initializer.WeightsInitializer
import estuary.components.regularizer.Regularizer
import estuary.components.support.{CanBackward, CanForward, CanSetParam}

trait Layer extends Serializable {
  val numHiddenUnits: Int

  def copyStructure: Layer

  def forward(yPrevious: DenseMatrix[Double])(implicit op: CanForward[ClassicLayer, DenseMatrix[Double], DenseMatrix[Double]]): DenseMatrix[Double]

  def forwardForPrediction(yPrevious: DenseMatrix[Double])(implicit op: CanForward[ClassicLayer, DenseMatrix[Double], DenseMatrix[Double]]): DenseMatrix[Double]

  def backward(dYCurrent: DenseMatrix[Double], regularizer: Option[Regularizer])(implicit op: CanBackward[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])]): (DenseMatrix[Double], DenseMatrix[Double])

  def init(initializer: WeightsInitializer): DenseMatrix[Double]

  def getReguCost(regularizer: Option[Regularizer]): Double

  def setParam(param: DenseMatrix[Double])(implicit op: CanSetParam[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseVector[Double], DenseVector[Double])],
                                           op2: CanSetParam[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseVector[Double])]): Unit
}
