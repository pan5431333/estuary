package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.ConvLayer.{ConvSize, Filter, calConvSize}
import estuary.components.layers.PoolingLayer.PoolType
import estuary.components.regularizer.Regularizer

class PoolingLayer(val poolSize: Int, val stride: Int, val pad: Int, val poolType: PoolType) extends Layer {
  protected var preConvSize: ConvSize = _

  lazy protected val filter: Filter = Filter(poolSize, pad, stride, preConvSize.channel, preConvSize.channel)
  lazy protected val outputConvSize: ConvSize = calConvSize(preConvSize, filter)

  lazy protected val maskMatrix: Array[DenseMatrix[Double]] = new Array[DenseMatrix[Double]](preConvSize.channel)

  def setPreConvSize(pre: ConvSize): this.type = {
    this.preConvSize = pre
    this
  }

  def setPreConvSize(preHeight: Int, preWidth: Int, preChannel: Int): this.type = {
    setPreConvSize(ConvSize(preHeight, preWidth, preChannel))
  }

  lazy override val numHiddenUnits: Int = outputConvSize.dataLength

  override def copyStructure: Layer = PoolingLayer(poolSize, stride, pad, poolType, preConvSize)

  override def forward(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    val preConvSizeChannel = ConvSize(preConvSize.height, preConvSize.width, 1)
    val filterChannel = Filter(poolSize, pad, stride, 1, 1)

    var resRow: Int = 0
    val pooledData = (for (c <- 0 until preConvSize.channel) yield {
      val startCol = c * (preConvSize.height * preConvSize.width)
      val endCol = (c + 1) * (preConvSize.height * preConvSize.width)
      val yPreviousChannel = yPrevious(::, startCol until endCol)
      val yPChannelCol = ConvLayer.im2col(yPreviousChannel, preConvSizeChannel, filterChannel)
      resRow = yPChannelCol.rows
      maskMatrix(c) = DenseMatrix.zeros[Double](yPChannelCol.rows, yPChannelCol.cols)

      for (i <- (0 until resRow).par) yield {
        val target = poolType.pool(yPChannelCol(i, ::).t)
        var maskVector = yPChannelCol(i, ::).t.map(d => if (d == target) 1.0 else 0.0)
        maskVector = maskVector / sum(maskVector)
        maskMatrix(c)(i, ::) := maskVector.t
        target
      }
    }).flatten.toArray

    val pooledMatrix = DenseMatrix.create[Double](resRow, preConvSize.channel, pooledData)
    ConvLayer.col2im(pooledMatrix, outputConvSize)
  }

  override def forwardForPrediction(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = forward(yPrevious)

  override def backward(dYCurrent: DenseMatrix[Double], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val dZCol = ConvLayer.imGrad2Col(dYCurrent, outputConvSize)

    val masks = for (c <- 0 until dZCol.cols) yield {
      val dZChannel = dZCol(::, c)
      val mask = maskMatrix(c)
      val dZChannelMatrix = dZChannel * DenseVector.ones[Double](mask.cols).t
      mask *:* dZChannelMatrix
    }

    val gradsMatrix = masks.reduceLeft[DenseMatrix[Double]]{ case (total, mask) => DenseMatrix.horzcat(total, mask)}

    val grads = ConvLayer.colGrad2Im(gradsMatrix, preConvSize, filter)
    (grads, DenseMatrix.zeros[Double](1,1))
  }

  override def init(initializer: WeightsInitializer): DenseMatrix[Double] = DenseMatrix.zeros[Double](1,1)

  override def getReguCost(regularizer: Option[Regularizer]): Double = 0.0

  override def setParam(param: DenseMatrix[Double]): Unit = {}
}

object PoolingLayer {
  def apply(poolSize: Int, stride: Int, pad: Int, poolType: PoolType, preConvSize: ConvSize): PoolingLayer = {
    new PoolingLayer(poolSize, stride, pad, poolType).setPreConvSize(preConvSize)
  }

  sealed trait PoolType {
    def pool(d: DenseVector[Double]): Double
  }

  object MAX_POOL extends PoolType {
    override def pool(d: DenseVector[Double]): Double = breeze.linalg.max(d)
  }

  object AVG_POOL extends PoolType {
    override def pool(d: DenseVector[Double]): Double = breeze.stats.mean(d)
  }

}
