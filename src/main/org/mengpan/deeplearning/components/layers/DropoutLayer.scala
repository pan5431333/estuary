package org.mengpan.deeplearning.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import org.mengpan.deeplearning.components.regularizer.Regularizer

/**
  * Created by mengpan on 2017/9/7.
  */
class DropoutLayer extends Layer{

  protected var dropoutRate: Double = _
  def setDropoutRate(dropoutRate: Double): this.type = {
    assert(dropoutRate <= 1 && dropoutRate >= 0, "dropout rate must be between 0 and 1")

    this.dropoutRate = dropoutRate
    this
  }

  protected var dropoutVector: DenseVector[Double] = _

  protected override def activationFuncEval(zCurrent: DenseMatrix[Double]):
  DenseMatrix[Double] = {
    dropoutVector = generateDropoutVector(numHiddenUnits, dropoutRate)

    val numExamples = zCurrent.rows
    val oneVector = DenseVector.ones[Double](numExamples)

    zCurrent *:* (oneVector * dropoutVector.t) / (1.0 - this.dropoutRate)
  }

  protected override def activationGradEval(zCurrent: DenseMatrix[Double]):
  DenseMatrix[Double] = {
    val numExamples = zCurrent.rows
    val oneVector = DenseVector.ones[Double](numExamples)

    (oneVector * dropoutVector.t) / (1.0 - this.dropoutRate)
  }

  private def generateDropoutVector(numHiddenUnits: Int, dropoutRate: Double):
  DenseVector[Double] = {
    DenseVector.rand[Double](this.numHiddenUnits)
      .map{i =>
        if (i <= this.dropoutRate) 0.0 else 1.0
      }
  }


  override def forward(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    this.yPrevious = yPrevious
    activationFuncEval(yPrevious)
  }

  override def backward(dYCurrent: DenseMatrix[Double], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val filterMat = activationGradEval(yPrevious)

    (dYCurrent *:* filterMat, DenseMatrix.zeros[Double](previousHiddenUnits+1, numHiddenUnits))
  }
}

object DropoutLayer {
  def apply(dropoutRate: Double): DropoutLayer = {
    new DropoutLayer()
      .setNumHiddenUnits(100)
      .setDropoutRate(dropoutRate)
      .setBatchNorm(false)
  }
}
