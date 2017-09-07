package org.mengpan.deeplearning.components.layers
import breeze.linalg.{DenseMatrix, DenseVector}
import org.mengpan.deeplearning.utils.MyDict

/**
  * Created by mengpan on 2017/8/26.
  */
class ReluLayer extends Layer{
  override var numHiddenUnits: Int = _
//  protected override var activationFunc: Byte = MyDict.ACTIVATION_RELU

  protected override def activationFuncEval(zCurrent: DenseMatrix[Double]):
  DenseMatrix[Double] = {
    zCurrent.map{i =>
      if (i >= 0) i else 0.0
    }
  }

  protected override def activationGradEval(zCurrent: DenseMatrix[Double]):
  DenseMatrix[Double] = {
    zCurrent.map{i =>
      if (i >= 0) 1.0 else 0.0
    }
  }
}

