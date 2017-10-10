package estuary.components.layers

import breeze.linalg.DenseMatrix

/**
  * Created by mengpan on 2017/8/26.
  */
class ReluLayer extends Layer {

  protected override def activationFuncEval(zCurrent: DenseMatrix[Double]):
  DenseMatrix[Double] = {
    zCurrent.map { i =>
      if (i >= 0) i else 0.0
    }
  }

  protected override def activationGradEval(zCurrent: DenseMatrix[Double]):
  DenseMatrix[Double] = {
    zCurrent.map { i =>
      if (i >= 0) 1.0 else 0.0
    }
  }
}

object ReluLayer {
  def apply(numHiddenUnits: Int, batchNorm: Boolean = true): ReluLayer = {
    new ReluLayer()
      .setNumHiddenUnits(numHiddenUnits)
      .setBatchNorm(batchNorm)
  }
}

