package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import estuary.components.layers.ConvLayer.{ConvSize, Filter, calConvSize}
import estuary.components.layers.LayerLike.ForPrediction
import estuary.components.layers.PoolingLayer.PoolType
import estuary.components.regularizer.Regularizer
import estuary.components.support._

class PoolingLayer(val poolSize: Int, val stride: Int, val pad: Int, val poolType: PoolType)
  extends Layer with LayerLike[PoolingLayer] {

  lazy val numHiddenUnits: Int = outputConvSize.dataLength

  var preConvSize: ConvSize = _
  lazy val filter: Filter = Filter(poolSize, pad, stride, preConvSize.channel, preConvSize.channel)
  lazy val outputConvSize: ConvSize = calConvSize(preConvSize, filter)
  lazy val maskMatrix: Array[DenseMatrix[Double]] = new Array[DenseMatrix[Double]](preConvSize.channel)

  def setPreConvSize(pre: ConvSize): this.type = {
    this.preConvSize = pre
    this
  }

  def setPreConvSize(preHeight: Int, preWidth: Int, preChannel: Int): this.type = {
    setPreConvSize(ConvSize(preHeight, preWidth, preChannel))
  }

  override def copyStructure: PoolingLayer = PoolingLayer(poolSize, stride, pad, poolType, preConvSize)
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

  implicit val poolingLayerCanSetParam: CanSetParam[PoolingLayer, None.type] = (_, _) => None

  implicit val poolingLayerCanExportParam: CanExportParam[PoolingLayer, None.type] = (_) => None

  implicit val poolingLayerCanAutoInit: CanAutoInit[PoolingLayer] = (_, _) => {}

  implicit val poolingLayerCanForward: CanForward[PoolingLayer, DenseMatrix[Double], DenseMatrix[Double]] =
    (input, by) => {
      val preConvSizeChannel = ConvSize(by.preConvSize.height, by.preConvSize.width, 1)
      val filterChannel = Filter(by.poolSize, by.pad, by.stride, 1, 1)

      var resRow: Int = 0
      val pooledData = (for (c <- 0 until by.preConvSize.channel) yield {
        val startCol = c * (by.preConvSize.height * by.preConvSize.width)
        val endCol = (c + 1) * (by.preConvSize.height * by.preConvSize.width)
        val yPreviousChannel = input(::, startCol until endCol)

        val yPChannelCol = implicitly[CanTransformForConv[TransformType.IMAGE_TO_COL, (DenseMatrix[Double], ConvSize, Filter), DenseMatrix[Double]]]
          .transform(yPreviousChannel, preConvSizeChannel, filterChannel)

        resRow = yPChannelCol.rows
        by.maskMatrix(c) = DenseMatrix.zeros[Double](yPChannelCol.rows, yPChannelCol.cols)

        for (i <- (0 until resRow).par) yield {
          val target = by.poolType.pool(yPChannelCol(i, ::).t)
          var maskVector = yPChannelCol(i, ::).t.map(d => if (d == target) 1.0 else 0.0)
          maskVector = maskVector / sum(maskVector)
          by.maskMatrix(c)(i, ::) := maskVector.t
          target
        }
      }).flatten.toArray

      val pooledMatrix = DenseMatrix.create[Double](resRow, by.preConvSize.channel, pooledData)
      val res = implicitly[CanTransformForConv[TransformType.COL_TO_IMAGE, (DenseMatrix[Double], ConvSize), DenseMatrix[Double]]]
        .transform(pooledMatrix, by.outputConvSize)
      res
    }

  implicit val poolingLayerCanForwardForPrediction: CanForward[PoolingLayer, ForPrediction[DenseMatrix[Double]], DenseMatrix[Double]] =
    (input, by) => by.forward(input.input)

  implicit val poolingLayerCanBackward: CanBackward[PoolingLayer, DenseMatrix[Double], (DenseMatrix[Double], None.type)] =
    (input, by, regularizer) => {
      val dZCol = implicitly[CanTransformForConv[TransformType.IMAGE_GRAD_2_COL, (DenseMatrix[Double], ConvSize), DenseMatrix[Double]]]
        .transform(input, by.outputConvSize)

      val masks = for (c <- 0 until dZCol.cols) yield {
        val dZChannel = dZCol(::, c)
        val mask = by.maskMatrix(c)
        val dZChannelMatrix = dZChannel * DenseVector.ones[Double](mask.cols).t
        mask *:* dZChannelMatrix
      }

      val gradsMatrix = masks.reduceLeft[DenseMatrix[Double]] { case (total, mask) => DenseMatrix.horzcat(total, mask) }

      val grads = implicitly[CanTransformForConv[TransformType.COL_GRAD_2_IMAGE, (DenseMatrix[Double], ConvSize, Filter), DenseMatrix[Double]]]
        .transform(gradsMatrix, by.preConvSize, by.filter)

      (grads, None)
    }

  implicit val poolingLayerCanRegularize = new CanRegularize[PoolingLayer] {
    override def regu(foor: PoolingLayer, regularizer: Option[Regularizer]): Double = 0.0
  }

}
