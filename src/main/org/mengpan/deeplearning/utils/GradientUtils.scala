package org.mengpan.deeplearning.utils

import breeze.linalg.DenseMatrix
import breeze.numerics.{pow, sigmoid}

/**
  * Created by mengpan on 2017/8/25.
  */
object GradientUtils {
  def reluGrad(z: DenseMatrix[Double]): DenseMatrix[Double] = {
    val numRows = z.rows
    val numCols = z.cols

    val res = DenseMatrix.zeros[Double](numRows, numCols)

    (0 until numRows).foreach{i =>
      (0 until numCols).foreach{j =>
        res(i, j) = if (z(i, j) >= 0) 1.0 else 0.0
      }
    }

    res
  }

  def tanhGrad(z: DenseMatrix[Double]): DenseMatrix[Double] = {
    1.0 - pow(z, 2)
  }

  def sigmoidGrad(z: DenseMatrix[Double]): DenseMatrix[Double] = {
    val res = sigmoid(z) *:* (1.0 - sigmoid(z))
    res.map{d =>
      if(d < pow(10.0, -9)) pow(10.0, -9)
      else if (d > pow(10.0, 2)) pow(10.0, 2)
      else d
    }
  }

  def getGradByFuncType(activationFuncType: Byte): DenseMatrix[Double] => DenseMatrix[Double] = {
    activationFuncType match {
      case MyDict.ACTIVATION_TANH => tanhGrad
      case MyDict.ACTIVATION_RELU => reluGrad
      case MyDict.ACTIVATION_SIGMOID => sigmoidGrad
      case _ => throw new Exception("Unsupported type of activation function")
    }
  }
}

