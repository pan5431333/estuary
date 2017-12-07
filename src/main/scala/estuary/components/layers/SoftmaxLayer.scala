package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import estuary.components.regularizer.Regularizer
import estuary.components.support._

/**
  * Created by mengpan on 2017/9/14.
  */
class SoftmaxLayer(val numHiddenUnits: Int, val batchNorm: Boolean) extends ClassicLayer with SoftmaxActivator{

  /** Make it more convenient to write code in type class design pattern */
  override def repr: SoftmaxLayer = this.asInstanceOf[SoftmaxLayer]

  def copyStructure: SoftmaxLayer = new SoftmaxLayer(numHiddenUnits, batchNorm).setPreviousHiddenUnits(previousHiddenUnits)

  override def backward[BackwardInput, BackwardOutput](dYCurrent: BackwardInput, regularizer: Option[Regularizer])
                                                      (implicit op: CanBackward[ClassicLayer, BackwardInput, BackwardOutput]): BackwardOutput =
    implicitly[CanBackward[SoftmaxLayer, BackwardInput, BackwardOutput]].backward(dYCurrent, repr, regularizer)
}

object SoftmaxLayer {
  def apply(numHiddenUnits: Int, batchNorm: Boolean = false): SoftmaxLayer = {
    new SoftmaxLayer(numHiddenUnits, batchNorm)
  }

  implicit val softmaxLayerCanBackward: CanBackward[SoftmaxLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])] =
    (input, by, regularizer) => {
      val dY = input
      val numExamples = dY.rows
      val n = numExamples.toDouble
      val w = by.param._1
      val b = by.param._2

      //HACK!
      val label = dY

      val dZ = by.y - label

      val dWCurrent = regularizer match {
        case None => by.yPrevious.t * dZ / n
        case Some(regu) => by.yPrevious.t * dZ / n
      }
      val dBCurrent = (DenseVector.ones[Double](numExamples).t * dZ).t / n

      val grads = DenseMatrix.vertcat(dWCurrent, dBCurrent.toDenseMatrix)
      (dZ * w.t, grads)
    }
}
