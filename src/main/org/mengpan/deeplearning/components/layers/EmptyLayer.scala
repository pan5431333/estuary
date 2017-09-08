package org.mengpan.deeplearning.components.layers
import breeze.linalg.DenseMatrix

/**
  * Created by mengpan on 2017/9/7.
  */
object EmptyLayer extends Layer{
  override var numHiddenUnits: Int = 0

  override protected def activationFuncEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] =
    throw new Error("EmptyLayer.activationFuncEval")

  override protected def activationGradEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] =
    throw new Error("EmptyLayer.activationGradEval")

}
