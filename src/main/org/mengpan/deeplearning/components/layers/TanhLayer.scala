package org.mengpan.deeplearning.components.layers
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{pow, tanh}
import org.mengpan.deeplearning.utils.MyDict

/**
  * Created by mengpan on 2017/8/26.
  */
class TanhLayer extends Layer{
  override var numHiddenUnits: Int = _
//  protected override var activationFunc: Byte = MyDict.ACTIVATION_TANH

  protected override def activationFuncEval(zCurrent: DenseMatrix[Double]):
  DenseMatrix[Double] = {
    tanh(zCurrent)
  }

  protected override def activationGradEval(zCurrent: DenseMatrix[Double]):
  DenseMatrix[Double] = {
    1.0 - pow(zCurrent, 2)
  }
}

