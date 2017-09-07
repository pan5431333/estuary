package org.mengpan.deeplearning.utils

import breeze.linalg.DenseMatrix
import breeze.numerics.{pow, relu, sigmoid, tanh}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.components.layers._

/**
  * Created by mengpan on 2017/8/26.
  */
object ActivationUtils {
  val logger = Logger.getLogger("ActivationUtils")

  def getLayerByActivationType(activationType: Byte): Layer = {
    activationType match {
      case MyDict.ACTIVATION_TANH => new TanhLayer()
      case MyDict.ACTIVATION_RELU => new ReluLayer()
      case MyDict.ACTIVATION_SIGMOID => new SigmoidLayer()
      case MyDict.ACTIVATION_DROPOUT => new DropoutLayer()
      case _ => throw new Exception("Unsupported type of activation function")
    }
  }
}
