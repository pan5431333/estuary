package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.LayerLike.ForPrediction
import estuary.components.regularizer.Regularizer
import estuary.components.support._

/**
  * Interface for neural network's layer.
  */
trait ClassicLayer extends Layer
  with LayerLike[ClassicLayer]
  with Activator {

  override def hasParams = true

  protected[estuary] var param: (DenseMatrix[Double], DenseVector[Double]) = _
  protected[estuary] var previousHiddenUnits: Int = _
  /** Cache processed data */
  protected[estuary] var yPrevious: DenseMatrix[Double] = _
  protected[estuary] var z: DenseMatrix[Double] = _
  protected[estuary] var y: DenseMatrix[Double] = _

  def setPreviousHiddenUnits(n: Int): this.type = {
    this.previousHiddenUnits = n
    this
  }

  override def toString: String =
    s"""
       |ClassicLayer: ${getClass.getSimpleName},
       |Number of Hidden Units: $numHiddenUnits,
       |Previous Number of Hidden Units? $previousHiddenUnits
    """.stripMargin
}

object ClassicLayer {

  implicit val classicLayerCanSetParam: CanSetParam[ClassicLayer, DenseMatrix[Double]] =
    (from, foor) => {
      val w = from(0 to from.rows - 2, ::)
      val b = from(from.rows - 1, ::).t
      foor.param = (w, b)
      (w, b)
    }

  implicit val classicLayerCanExportParam: CanExportParam[ClassicLayer, DenseMatrix[Double]] =
    (from) => {
      val w = from.param._1
      val b = from.param._2
      DenseMatrix.vertcat(w, b.toDenseMatrix)
    }

  implicit val classicLayerCanAutoInit: CanAutoInit[ClassicLayer] =
    (foor: ClassicLayer, initializer: WeightsInitializer) => {
      val w = initializer.init(foor.previousHiddenUnits, foor.numHiddenUnits)
      val b = DenseVector.zeros[Double](foor.numHiddenUnits)
      foor.param = (w, b)
    }

  implicit val classicLayerCanForward: CanForward[ClassicLayer, DenseMatrix[Double], DenseMatrix[Double]] =
    (input, by) => {
      val w = by.param._1
      val b = by.param._2
      by.yPrevious = input
      val numExamples = input.rows
      by.z = input * w + DenseVector.ones[Double](numExamples) * b.t
      by.y = by.activate(by.z)
      by.y
    }

  implicit val classicLayerCanForwardForPrediction: CanForward[ClassicLayer, ForPrediction[DenseMatrix[Double]], DenseMatrix[Double]] =
    (input, by) => {
      by.forward(input.input)
    }

  implicit val classicLayerCanBackward: CanBackward[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])] =
    (input, by, regularizer) => {
      val w = by.param._1
      val b = by.param._2
      val numExamples = input.rows
      val n = numExamples.toDouble

      val dZ = input *:* by.activateGrad(by.z)
      val dWCurrent = regularizer match {
        case None => by.yPrevious.t * dZ / n
        case Some(regu) => by.yPrevious.t * dZ / n
      }
      val dBCurrent = (DenseVector.ones[Double](numExamples).t * dZ).t / numExamples.toDouble
      val dYPrevious = dZ * w.t

      val grads = DenseMatrix.vertcat(dWCurrent, dBCurrent.toDenseMatrix)

      (dYPrevious, grads)
    }

  implicit val classicLayerCanRegularize = new CanRegularize[ClassicLayer] {
    override def regu(foor: ClassicLayer, regularizer: Option[Regularizer]): Double = {
      regularizer match {
        case None => 0.0
        case r: Regularizer => r.getReguCost(foor.param._1)
      }
    }
  }


}
