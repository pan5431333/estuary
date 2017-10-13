package estuary.components.layers

import breeze.linalg.DenseMatrix

/**
  * Created by mengpan on 2017/8/26.
  */
class ReluLayer extends Layer {

  protected def activationFuncEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    val res = DenseMatrix.zeros[Double](zCurrent.rows, zCurrent.cols)
    for {i <- (0 until zCurrent.rows).par
         j <- (0 until zCurrent.cols).par
    } {
      res(i, j) = if (zCurrent(i, j) >= 0) zCurrent(i, j) else 0.0
    }
    res
  }

  protected def activationGradEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    val res = DenseMatrix.zeros[Double](zCurrent.rows, zCurrent.cols)

    for {i <- (0 until zCurrent.rows).par
         j <- (0 until zCurrent.cols).par
    } {
      res(i, j) = if (zCurrent(i, j) >= 0) 1.0 else 0.0
    }
    res
  }

  def copyStructure: ReluLayer = new ReluLayer().setPreviousHiddenUnits(previousHiddenUnits).setNumHiddenUnits(numHiddenUnits).setBatchNorm(batchNorm)
}

object ReluLayer {
  def apply(numHiddenUnits: Int, batchNorm: Boolean = true): ReluLayer = {
    new ReluLayer()
      .setNumHiddenUnits(numHiddenUnits)
      .setBatchNorm(batchNorm)
  }
}

