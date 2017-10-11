package estuary.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector, min}

import scala.util.Random

/**
  * Optimizer using mini-batch optimization algprithm.
  * Its implementations has the ability of "getMiniBatches()"
  */
trait MiniBatchable {
  protected var miniBatchSize: Int = _

  def setMiniBatchSize(miniBatchSize: Int): this.type = {
    assert(miniBatchSize > 0, "Minibatch size must be positive. ")

    this.miniBatchSize = miniBatchSize
    this
  }

  /**
    * Split the whole training set (feature, label) to an iterator of multiple mini-batches.
    *
    * @param feature DenseMatrix of shape (n, p) where n: number of training examples,
    *                p: dimension of input feature.
    * @param label   DenseMatrix of shape (n, q) where n: number of training examples,
    *                q: number of distinct labels.
    * @return An iterator of multiple minibatches.
    */
  def getMiniBatches(feature: DenseMatrix[Double],
                     label: DenseMatrix[Double]): Iterator[(DenseMatrix[Double], DenseMatrix[Double])] = {
    assert(feature.rows == label.rows, "feature.rows != label.rows")

    this.miniBatchSize match {
      case a if a > feature.rows => throw new IllegalArgumentException(
        "mini batch size(" + this.miniBatchSize + ")must be less than number of examples(" + feature.rows + ")!")
      case a if a == feature.rows => Iterator((feature, label))
      case a if a > 0 => getPositiveNumMiniBatches(feature, label, a)
      case _ => throw new IllegalArgumentException("mini-batch size: " + this.miniBatchSize + " number of exmaples: " + feature.rows)
    }
  }

  /**
    * Split up a training set into an iterator of multiple mini-batches. Each mini-batch
    * has miniBatchSize# of examples.
    *
    * @param feature DenseMatrix of shape (n, p) where n: number of training examples,
    *                p: dimension of input feature.
    * @param label   DenseMatrix of shape (n, q) where n: number of training examples,
    *                q: number of distinct labels.
    * @param miniBatchSize
    * @return An iterator of multiple minibatches.
    */
  private def getPositiveNumMiniBatches(feature: DenseMatrix[Double], label: DenseMatrix[Double], miniBatchSize: Int): Iterator[(DenseMatrix[Double], DenseMatrix[Double])] = {
    val numExamples = feature.rows
    val shuffledIndex = Random.shuffle[Int, Vector]((0 until numExamples).toVector)
    val numMiniBatchesFloor = numExamples / miniBatchSize
    val isDivided = numExamples % miniBatchSize == 0
    val numMiniBatches = if (isDivided) numMiniBatchesFloor else numMiniBatchesFloor + 1

    (0 until numMiniBatches).toIterator.map { i =>
      val startIndex = i * miniBatchSize
      val endIndex = min((i + 1) * miniBatchSize, numExamples)
      val indexes = shuffledIndex.slice(startIndex, endIndex)
      (feature(indexes, ::).toDenseMatrix, label(indexes, ::).toDenseMatrix)
    }
  }
}
