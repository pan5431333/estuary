package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.exp

trait SoftmaxActivator extends Activator{
  /**
    * Scale input zCurrent for each ROW, since each row represents outputs of
    * one training exmaple.
    *
    * @param zCurrent of shape (numExamples, numHiddenUnitsOutputLayer)
    * @return of shape (numExamples, output)
    */
  def activate(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {

    val res = DenseMatrix.zeros[Double](zCurrent.rows, zCurrent.cols)
    for (i <- 0 until zCurrent.rows) {
      res(i, ::) := softMaxScale(zCurrent(i, ::).t).t
    }
    res
  }

  /**
    * Since backward has been overriden, this method will no longer be needed
    */
  def activateGrad(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = ???

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
