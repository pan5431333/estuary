package org.mengpan.deeplearning.components.layers
import breeze.linalg.{DenseMatrix, DenseVector, min, softmax, sum}
import breeze.numerics.{exp, pow}

/**
  * Created by mengpan on 2017/9/14.
  */
class SoftmaxLayer extends Layer{


  override def backward(dYCurrent: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    this.batchNorm match {
      case true => backwardWithBatchNorm(dYCurrent)
      case _ => backwardWithoutBatchNorm(dYCurrent)
    }


  }

  private def backwardWithoutBatchNorm(dYCurrent: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val numExamples = dYCurrent.rows

    //HACK!
    val label = dYCurrent

    val dZCurrent = y - label

    val dWCurrent = yPrevious.t * dZCurrent / numExamples.toDouble
    val dBCurrent = (DenseVector.ones[Double](numExamples).t * dZCurrent).t /
      numExamples.toDouble

    val grads = DenseMatrix.vertcat(dWCurrent, dBCurrent.toDenseMatrix)
    (dZCurrent * w.t, grads)
  }

  private def backwardWithBatchNorm(dYCurrent: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val numExamples = dYCurrent.rows
    val oneVector = DenseVector.ones[Double](numExamples)

    //HACK!
    val label = dYCurrent

    val dZDelta = y - label
    val dZNorm = dZDelta *:* (oneVector * beta.t)
    val dAlpha = dZDelta.t * oneVector / numExamples.toDouble
    val dBeta = (dZDelta *:* zNorm).t * oneVector / numExamples.toDouble

    val dZ = DenseMatrix.zeros[Double](z.rows, z.cols)
    for (j <- 0 until z.cols) {
      val dZNormJ = dZNorm(::, j)
      val dZJ = (DenseMatrix.eye[Double](dZNormJ.length) / currentStddevZ(j) - DenseMatrix.ones[Double](dZNormJ.length, dZNormJ.length) / (numExamples.toDouble * currentStddevZ(j)) - (z(::, j) - currentMeanZ(j)) * (z(::, j) - currentMeanZ(j)).t / (numExamples.toDouble * pow(currentStddevZ(j), 3.0))) * dZNormJ
      dZ(::, j) := dZJ
    }

    val dWCurrent = yPrevious.t * dZ / numExamples.toDouble
    val dYPrevious = dZ * w.t

    val grads = DenseMatrix.vertcat(dWCurrent, dAlpha.toDenseMatrix, dBeta.toDenseMatrix)

    (dYPrevious, grads)
  }

  /**
    * Scale input zCurrent for each ROW, since each row represents outputs of
    * one training exmaple.
    *
    * @param zCurrent of shape (numExamples, numHiddenUnitsOutputLayer)
    * @return of shape (numExamples, output)
    */
  override protected def activationFuncEval(zCurrent: DenseMatrix[Double]):
  DenseMatrix[Double] = {

    val res = DenseMatrix.zeros[Double](zCurrent.rows, zCurrent.cols)
    for (i <- 0 until zCurrent.rows) {
      res(i, ::) := softMaxScale(zCurrent(i, ::).t).t
    }
    res
  }

  /**
    * Since backward has been overriden, this method will no longer be needed
    * @param zCurrent
    * @return
    */
  override protected def activationGradEval(zCurrent: DenseMatrix[Double]):
  DenseMatrix[Double] = ???

  /**
    * Scale by softmax for a vector, i.e. (a1, a2, a3) => summation = e^a1 + e^a2 + e^a3
    * => (e^a1/summation, e^a2/summation, e^a3/summation)
    * @param output
    * @return
    **/
  private def softMaxScale(_x: DenseVector[Double]): DenseVector[Double] = {
    //numerical stability 1
//    val x = _x - breeze.linalg.max(_x)
//    val exped = breeze.numerics.exp(x)
//    val summation = sum(exped)
//    exped.map{i =>
//      val t = i/summation
//      t
//    }

    //numerical stability 2 (Better!)
    val x = _x
    val exped = breeze.numerics.exp(x)
    val summation = sum(exped)
    val logsum = math.log(summation)
    val logged = x.map(i => i - logsum)
    exp(logged)
  }
}

object SoftmaxLayer {
  def apply(batchNorm: Boolean = false): SoftmaxLayer = {
    new SoftmaxLayer()
      .setBatchNorm(batchNorm)
      .setNumHiddenUnits(1)
  }
}
