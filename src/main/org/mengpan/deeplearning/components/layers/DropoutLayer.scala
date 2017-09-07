package org.mengpan.deeplearning.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import org.mengpan.deeplearning.utils.MyDict

/**
  * Created by mengpan on 2017/9/7.
  */
class DropoutLayer extends Layer{
  override var numHiddenUnits: Int = _
  protected override var activationFunc: Byte = MyDict.ACTIVATION_DROPOUT

  protected var dropoutRate: Double = _

  def setDropoutRate(dropoutRate: Double): this.type = {
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

    zCurrent *:* dropoutMatrix / (1.0 - this.dropoutRate)
  }

  private def generateDropoutVector(numHiddenUnits: Int, dropoutRate: Double):
  DenseVector[Double] = {
    DenseVector.rand[Double](this.numHiddenUnits)
      .map{i =>
        if (i <= this.dropoutRate) 0.0 else 1.0
      }
  }
}
