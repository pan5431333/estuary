package estuary.components.layers
import breeze.linalg.{DenseMatrix, DenseVector}
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.ConvLayer.{ConvSize, Filter, calConvSize}
import estuary.components.layers.PoolingLayer.PoolType
import estuary.components.regularizer.Regularizer

class PoolingLayer(val poolSize: Int, val stride: Int, val pad: Int, val poolType: PoolType) extends Layer{
  protected var preConvSize: ConvSize = _

  lazy protected val filter: Filter = Filter(poolSize, pad, stride, preConvSize.channel, preConvSize.channel)
  lazy protected val outputConvSize: ConvSize = calConvSize(preConvSize, filter)

  protected var maskMatrix: Array[DenseMatrix[Double]] = new Array[DenseMatrix[Double]](preConvSize.channel)

  def setPreConvSize(pre: ConvSize): this.type = {
    this.preConvSize = pre
    this
  }

  def setPreConvSize(preHeight: Int, preWidth: Int, preChannel: Int): this.type = {
    setPreConvSize(ConvSize(preHeight, preWidth, preChannel))
  }

  override val numHiddenUnits: Int = 0

  override def copyStructure: Layer = PoolingLayer(filter, poolType)

  override def forward(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    val preConvSizeChannel = ConvSize(preConvSize.height, preConvSize.width, 1)
    val filterChannel = Filter(poolSize, pad, stride, 1, 1)

    for (c <- 0 until preConvSize.channel) {
      val startCol = c * (preConvSize.height * preConvSize.width)
      val endCol = (c + 1) * (preConvSize.height * preConvSize.width)
      val yPreviousChannel = yPrevious(::, startCol until endCol)
      val yPChannelCol = ConvLayer.im2col(yPreviousChannel, preConvSizeChannel, filterChannel)

      maskMatrix(c) = DenseMatrix.zeros[Double](yPChannelCol.rows, yPChannelCol.cols)
    }


    val yPreviousCol = ConvLayer.im2col(yPrevious, preConvSize, filter)
    val pooled = DenseMatrix.zeros[Double](yPreviousCol.rows, 1)
    maskMatrix = DenseMatrix.zeros[Double](yPreviousCol.rows, yPreviousCol.cols)
    for (i <- 0 until pooled.rows) {
      val target = poolType.pool(yPreviousCol(i, ::).t)
      pooled(i, 0) = target
      var maskVector = yPreviousCol(i, ::).t.map( d => if (d == target) 1.0 else 0.0)
      maskVector = maskVector / breeze.linalg.sum(maskVector)
      maskMatrix(i, ::) := maskVector.t
    }
    ConvLayer.col2im(pooled, outputConvSize)
  }

  override def forwardForPrediction(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = forward(yPrevious)

  override def backward(dYCurrent: DenseMatrix[Double], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val dZCol = ConvLayer.imGrad2Col(dYCurrent, outputConvSize)
    val maskGrads = dZCol * DenseVector.ones[Double](dZCol.rows)
  }

  override def init(initializer: WeightsInitializer): DenseMatrix[Double] =
    throw new UnsupportedOperationException("Call init() on pooling layer")

  override def getReguCost(regularizer: Option[Regularizer]): Double =
    throw new UnsupportedOperationException("Call getReguCost() on pooling layer")

  override def setParam(param: DenseMatrix[Double]): Unit =
    throw new UnsupportedOperationException("Call setParam() on pooling layer")
}

object PoolingLayer {
  def apply(filter: Filter, poolType: PoolType): PoolingLayer = {
    new PoolingLayer(filter, poolType)
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
