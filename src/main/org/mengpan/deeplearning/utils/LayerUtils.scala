package org.mengpan.deeplearning.utils

import org.mengpan.deeplearning.layers.{Layer, ReluLayer, SigmoidLayer, TanhLayer}

/**
  * Created by mengpan on 2017/8/26.
  */
object LayerUtils {
  def getLayerByActivationType(activationType: Byte): Layer = {
    activationType match {
      case MyDict.ACTIVATION_TANH => new TanhLayer()
      case MyDict.ACTIVATION_RELU => new ReluLayer()
      case MyDict.ACTIVATION_SIGMOID => new SigmoidLayer()
      case _ => throw new Exception("Unsupported type of activation function")
    }
  }
}
