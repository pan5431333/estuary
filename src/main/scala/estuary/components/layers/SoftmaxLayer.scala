package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.exp
import estuary.components.regularizer.Regularizer

/**
  * Created by mengpan on 2017/9/14.
  */
class SoftmaxLayer(val numHiddenUnits: Int, val batchNorm: Boolean) extends ClassicLayer with SoftmaxActivator{

  def copyStructure: SoftmaxLayer = new SoftmaxLayer(numHiddenUnits, batchNorm).setPreviousHiddenUnits(previousHiddenUnits).asInstanceOf[SoftmaxLayer]

  override def backward(dYCurrent: DenseMatrix[Double], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    if (this.batchNorm) backwardWithBatchNorm(dYCurrent, yPrevious, regularizer)
    else backwardWithoutBatchNorm(dYCurrent, yPrevious, regularizer)
  }

  private def backwardWithoutBatchNorm(dYCurrent: DenseMatrix[Double], yPrevious: DenseMatrix[Double], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val numExamples = dYCurrent.rows
    val n = numExamples.toDouble

    //HACK!
    val label = dYCurrent

    val dZ = y - label

    val dWCurrent = regularizer match {
      case None => yPrevious.t * dZ / n
      case Some(regu) => yPrevious.t * dZ / n + regu.getReguCostGrad(w)
    }
    val dBCurrent = (DenseVector.ones[Double](numExamples).t * dZ).t / n

    val grads = DenseMatrix.vertcat(dWCurrent, dBCurrent.toDenseMatrix)
    (dZ * w.t, grads)
  }


  private def backwardWithBatchNorm(dYCurrent: DenseMatrix[Double], yPrevious: DenseMatrix[Double], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val numExamples = dYCurrent.rows
    val n = numExamples.toDouble
    val oneVector = DenseVector.ones[Double](numExamples)

    //HACK!
    val label = dYCurrent

    val dZDelta = y - label
    val dZNorm = dZDelta *:* (oneVector * beta.t)
    val dAlpha = dZDelta.t * oneVector / numExamples.toDouble
    val dBeta = (dZDelta *:* zNorm).t * oneVector / numExamples.toDouble

    //  val dZ = normalizeGrad(dZNorm, z, currentMeanZ, currentStddevZ)
    val dZ = normalizeGradVec(dZNorm, z, currentMeanZ, currentStddevZ)

    val dWCurrent = regularizer match {
      case None => yPrevious.t * dZ / n
      case Some(regu) => yPrevious.t * dZ / n + regu.getReguCostGrad(w)
    }
    val dYPrevious = dZ * w.t

    val grads = DenseMatrix.vertcat(dWCurrent, dAlpha.toDenseMatrix, dBeta.toDenseMatrix)

    (dYPrevious, grads)
  }
}

object SoftmaxLayer {
  def apply(numHiddenUnits: Int, batchNorm: Boolean = false): SoftmaxLayer = {
    new SoftmaxLayer(numHiddenUnits, batchNorm)
  }
}
