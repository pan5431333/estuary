package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}

trait DropoutActivator extends Activator{
  val dropoutRate: Double

  protected var dropoutVector: DenseVector[Double] = _

  private def generateDropoutVector(numHiddenUnits: Int, dropoutRate: Double): DenseVector[Double] = {
    val randVec = DenseVector.rand[Double](numHiddenUnits)

    val res = DenseVector.zeros[Double](randVec.length)
    for (i <- (0 until randVec.length).par) {
      res(i) = if (randVec(i) <= this.dropoutRate) 0.0 else 1.0
    }
    res
  }

  protected def activate(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    dropoutVector = generateDropoutVector(zCurrent.cols, dropoutRate)

    val numExamples = zCurrent.rows
    val oneVector = DenseVector.ones[Double](numExamples)
    zCurrent *:* (oneVector * dropoutVector.t) / (1.0 - this.dropoutRate)
  }

  protected def activateGrad(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    val numExamples = zCurrent.rows
    val oneVector = DenseVector.ones[Double](numExamples)

    (oneVector * dropoutVector.t) / (1.0 - this.dropoutRate)
  }
}
