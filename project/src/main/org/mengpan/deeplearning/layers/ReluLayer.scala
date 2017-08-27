package org.mengpan.deeplearning.layers
import breeze.linalg.{DenseMatrix, DenseVector}
import org.mengpan.deeplearning.utils.MyDict

/**
  * Created by mengpan on 2017/8/26.
  */
class ReluLayer extends Layer{
  override var numHiddenUnits: Int = _
  override var activationFunc: Byte = MyDict.ACTIVATION_RELU
}

