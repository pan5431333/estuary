package org.mengpan.deeplearning.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import org.mengpan.deeplearning.utils.ResultUtils.{BackwardRes, ForwardRes}
import org.mengpan.deeplearning.utils.{MyDict, ResultUtils}

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

  protected lazy val dropoutVector: DenseVector[Double] = generateDropoutVector(numHiddenUnits, dropoutRate)

  protected override def activationFuncEval(zCurrent: DenseMatrix[Double]):
  DenseMatrix[Double] = {
    val dropoutMatrix = DenseMatrix.tabulate[Double](zCurrent.rows, this.dropoutVector.length){
      (i, j) => this.dropoutVector(j)
    }

    dropoutMatrix *:* zCurrent / (1.0 - this.dropoutRate)
  }

  protected override def activationGradEval(zCurrent: DenseMatrix[Double]):
  DenseMatrix[Double] = {
    val dropoutMatrix = DenseMatrix.tabulate[Double](zCurrent.rows, this.dropoutVector.length){
      (i, j) => this.dropoutVector(j)
    }

    dropoutMatrix / (1.0 - this.dropoutRate)
  }

  private def generateDropoutVector(numHiddenUnits: Int, dropoutRate: Double):
  DenseVector[Double] = {
    DenseVector.rand[Double](this.numHiddenUnits)
      .map{i =>
        if (i <= this.dropoutRate) 0.0 else 1.0
      }
  }

  override def forward(yPrevious: DenseMatrix[Double],
                       w: DenseMatrix[Double],
                       b: DenseVector[Double]):
  ResultUtils.ForwardRes = {
    val zCurrent = yPrevious
    val yCurrent = activationFuncEval(zCurrent)
    new ForwardRes(yPrevious, zCurrent, yCurrent)
  }

  override def backward(dYCurrent: DenseMatrix[Double],
                        forwardRes: ResultUtils.ForwardRes,
                        w: DenseMatrix[Double],
                        b: DenseVector[Double]):
  ResultUtils.BackwardRes = {

    val zCurrent = forwardRes.zCurrent
    val dZCurrent = dYCurrent *:* activationGradEval(zCurrent)
    val dYPrevious = dZCurrent
    val dW = null
    val dB = null
    new BackwardRes(dYPrevious, dW, dB)
  }
}

object DropoutLayer {
  def apply(dropoutRate: Double): DropoutLayer = {
    new DropoutLayer()
      .setNumHiddenUnits(100)
      .setDropoutRate(dropoutRate)
  }
}
