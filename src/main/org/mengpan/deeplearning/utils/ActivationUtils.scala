package org.mengpan.deeplearning.utils

import breeze.linalg.DenseMatrix
import breeze.numerics.{relu, sigmoid, tanh}
import org.apache.log4j.Logger

/**
  * Created by mengpan on 2017/8/26.
  */
object ActivationUtils {
  val logger = Logger.getLogger("ActivationUtils")

  def getActivationFunc(activationFuncType: Byte): DenseMatrix[Double] => DenseMatrix[Double] = {
    activationFuncType match {
      case MyDict.ACTIVATION_SIGMOID => sigmoid(_: DenseMatrix[Double])
      case MyDict.ACTIVATION_TANH => tanh(_: DenseMatrix[Double])
      case MyDict.ACTIVATION_RELU => relu(_: DenseMatrix[Double])
      case _ => logger.fatal("Wrong hidden activation function param given, use tanh by default")
        tanh(_: DenseMatrix[Double])
    }
  }
}
