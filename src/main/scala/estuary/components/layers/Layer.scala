package estuary.components.layers
import breeze.linalg.{DenseMatrix, DenseVector, sum}
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.ConvLayer.{ConvSize, Filter, calConvSize}
import estuary.components.layers.LayerLike.ForPrediction
import estuary.components.layers.PoolingLayer.PoolType
import estuary.components.regularizer.Regularizer
import estuary.components.support._

sealed trait Layer extends LayerLike[Layer]{
  val numHiddenUnits: Int

  /**Used to distribute model instances onto multiple machines for distributed optimization*/
  def copyStructure: Layer
}

/**
  * Interface for neural network's layer.
  */
trait ClassicLayer extends Layer
  with LayerLike[ClassicLayer]
  with Activator with Trainable{

  protected[estuary] var param: (DenseMatrix[Double], DenseVector[Double]) = _
  protected[estuary] var previousHiddenUnits: Int = _
  /** Cache processed data */
  protected[estuary] var yPrevious: DenseMatrix[Double] = _
  protected[estuary] var z: DenseMatrix[Double] = _
  protected[estuary] var y: DenseMatrix[Double] = _

  def setPreviousHiddenUnits(n: Int): this.type = {
    this.previousHiddenUnits = n
    this
  }

  override def toString: String =
    s"""
       |ClassicLayer: ${getClass.getSimpleName},
       |Number of Hidden Units: $numHiddenUnits,
       |Previous Number of Hidden Units? $previousHiddenUnits
    """.stripMargin
}

object ClassicLayer {

  implicit val classicLayerCanSetParam: CanSetParam[ClassicLayer, DenseMatrix[Double]] =
    (from, foor) => {
      val w = from(0 to from.rows - 2, ::)
      val b = from(from.rows - 1, ::).t
      foor.param = (w, b)
      (w, b)
    }

  implicit val classicLayerCanExportParam: CanExportParam[ClassicLayer, DenseMatrix[Double]] =
    (from) => {
      val w = from.param._1
      val b = from.param._2
      DenseMatrix.vertcat(w, b.toDenseMatrix)
    }

  implicit val classicLayerCanAutoInit: CanAutoInit[ClassicLayer] =
    (foor: ClassicLayer, initializer: WeightsInitializer) => {
      val w = initializer.init(foor.previousHiddenUnits, foor.numHiddenUnits)
      val b = DenseVector.zeros[Double](foor.numHiddenUnits)
      foor.param = (w, b)
    }

  implicit val classicLayerCanForward: CanForward[ClassicLayer, DenseMatrix[Double], DenseMatrix[Double]] =
    (input, by) => {
      val w = by.param._1
      val b = by.param._2
      by.yPrevious = input
      val numExamples = input.rows
      by.z = input * w + DenseVector.ones[Double](numExamples) * b.t
      by.y = by.activate(by.z)
      by.y
    }

  implicit val classicLayerCanForwardForPrediction: CanForward[ClassicLayer, ForPrediction[DenseMatrix[Double]], DenseMatrix[Double]] =
    (input, by) => {
      by.forward(input.input)
    }

  implicit val classicLayerCanBackward: CanBackward[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])] =
    (input, by, regularizer) => {
      val w = by.param._1
      val b = by.param._2
      val numExamples = input.rows
      val n = numExamples.toDouble

      val dZ = input *:* by.activateGrad(by.z)
      val dWCurrent = regularizer match {
        case None => by.yPrevious.t * dZ / n
        case Some(regu) => by.yPrevious.t * dZ / n
      }
      val dBCurrent = (DenseVector.ones[Double](numExamples).t * dZ).t / numExamples.toDouble
      val dYPrevious = dZ * w.t

      val grads = DenseMatrix.vertcat(dWCurrent, dBCurrent.toDenseMatrix)

      (dYPrevious, grads)
    }

  implicit val classicLayerCanRegularize = new CanRegularize[ClassicLayer] {
    override def regu(foor: ClassicLayer, regularizer: Option[Regularizer]): Double = {
      regularizer match {
        case None => 0.0
        case r: Regularizer => r.getReguCost(foor.param._1)
      }
    }
  }
}


trait ConvLayer extends Layer with LayerLike[ConvLayer] with Activator with Trainable{

  val param: Filter
  lazy val numHiddenUnits: Int = outputConvSize.dataLength
  lazy val previousHiddenUnits: Int = preConvSize.dataLength

  protected[estuary] var preConvSize: ConvSize = _
  protected[estuary] lazy val outputConvSize: ConvSize = calConvSize(preConvSize, param)
  /** cache intermediate results to be used later */
  protected[estuary] var yPrevious: DenseMatrix[Double] = _
  protected[estuary] var yPreviousIm2Col: DenseMatrix[Double] = _
  protected[estuary] var z: DenseMatrix[Double] = _
  protected[estuary] var zIm2Col: DenseMatrix[Double] = _
  protected[estuary] var y: DenseMatrix[Double] = _
  protected[estuary] var filterMatrix: DenseMatrix[Double] = _
  protected[estuary] var filterBias: DenseVector[Double] = _

  def setPreConvSize(pre: ConvSize): this.type = {
    this.preConvSize = pre
    this
  }
  def setPreConvSize(preHeight: Int, preWidth: Int, preChannel: Int): this.type = {
    setPreConvSize(ConvSize(preHeight, preWidth, preChannel))
  }
}

object ConvLayer {

  def checkInputValidity(yPrevious: DenseMatrix[Double], pre: ConvLayer.ConvSize): Unit = {
    val cols = yPrevious.cols
    val convSize = pre.height * pre.width * pre.channel
    assert(cols == convSize, s"Input data's cols not equal to convSize, ($cols != ${pre.height} * ${pre.width} * ${pre.channel})")
  }

  protected def calOutDimension(inputDim: Int, filterSize: Int, pad: Int, stride: Int): Int = {
    (inputDim + 2 * pad - filterSize) / stride + 1
  }

  def calConvSize(pre: ConvLayer.ConvSize, filter: ConvLayer.Filter): ConvSize = {
    val outHeight = calOutDimension(pre.height, filter.size, filter.pad, filter.stride)
    val outWidth = calOutDimension(pre.width, filter.size, filter.pad, filter.stride)
    val outChannel = filter.newChannel
    ConvSize(outHeight, outWidth, outChannel)
  }

  case class Filter(size: Int, pad: Int, stride: Int, oldChannel: Int, newChannel: Int) {
    var w: DenseMatrix[Double] = DenseMatrix.zeros[Double](size * size * oldChannel, newChannel)
    var b: DenseVector[Double] = DenseVector.zeros[Double](newChannel)
    val matrixShape: (Int, Int) = (w.rows + 1, w.cols)

    def toIm2Col: (DenseMatrix[Double], DenseVector[Double]) =
      implicitly[CanTransformForConv[TransformType.FILTER_TO_COL, Filter, (DenseMatrix[Double], DenseVector[Double])]].transform(this)

    override def toString: String = {
      s"""(size: $size, pad: $pad, stride: $stride, oldChannel: $oldChannel, newChannel: $newChannel)"""
    }
  }

  case class ConvSize(height: Int, width: Int, channel: Int) {
    val dataLength: Int = height * width * channel

    def linearIndex(h: Int, w: Int, c: Int): Int = c * height * width + w * height + h

    def ==(that: ConvSize): Boolean = (height == that.height) && (width == that.width) && (channel == that.channel)

    def contains(height: Int, width: Int, channel: Int): Boolean =
      height <= this.height && width <= this.width && channel <= this.channel

    override def toString: String = s"""(height: $height, width: $width, channel: $channel)"""
  }

  implicit val convLayerCanSetParam: CanSetParam[ConvLayer, DenseMatrix[Double]] =
    (from, foor) => {
      val w = from(0 to from.rows - 2, ::)
      val b = from(from.rows - 1, ::).t
      foor.param.w = w
      foor.param.b = b
    }

  implicit val convLayerCanExportParam: CanExportParam[ConvLayer, DenseMatrix[Double]] =
    (from) => {
      DenseMatrix.vertcat(from.param.w, from.param.b.toDenseMatrix)
    }

  implicit val convLayerCanAutoInit: CanAutoInit[ConvLayer] =
    (foor: ConvLayer, initializer: WeightsInitializer) => {
      val w = initializer.init(foor.param.w.rows, foor.param.matrixShape._2)
      val b = DenseVector.zeros[Double](foor.param.matrixShape._2)
      foor.setParam(DenseMatrix.vertcat(w, b.toDenseMatrix))
    }

  implicit val convLayerCanForward: CanForward[ConvLayer, DenseMatrix[Double], DenseMatrix[Double]] =
    (input, by) => {
      ConvLayer.checkInputValidity(input, by.preConvSize)

      by.yPreviousIm2Col = implicitly[CanTransformForConv[TransformType.IMAGE_TO_COL, (DenseMatrix[Double], ConvSize, Filter), DenseMatrix[Double]]]
        .transform((input, by.preConvSize, by.param))

      val filterWeightsAndBias = by.param.toIm2Col
      by.filterMatrix = filterWeightsAndBias._1 // shape (size * size * oldChannel, newChannel)
      by.filterBias = filterWeightsAndBias._2

      by.zIm2Col = by.yPreviousIm2Col * by.filterMatrix + DenseVector.ones[Double](by.yPreviousIm2Col.rows) * by.filterBias.t
      by.z = implicitly[CanTransformForConv[TransformType.COL_TO_IMAGE, (DenseMatrix[Double], ConvSize), DenseMatrix[Double]]]
        .transform(by.zIm2Col, by.outputConvSize)
      by.y = by.activate(by.z)
      by.y
    }

  implicit val convLayerCanBackward: CanBackward[ConvLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])] =
    (input, by, regularizer) => {
      val dZ = input *:* by.activateGrad(by.z)

      val dZCol = implicitly[CanTransformForConv[TransformType.IMAGE_GRAD_2_COL, (DenseMatrix[Double], ConvSize), DenseMatrix[Double]]]
        .transform(dZ, by.outputConvSize)

      val dYPrevious = dZCol * by.filterMatrix.t
      val n = input.rows.toDouble
      val dW = by.yPreviousIm2Col.t * dZCol / n
      val dB = (dZCol.t * DenseVector.ones[Double](dZCol.rows)).toDenseMatrix / n
      val grads = DenseMatrix.vertcat(dW, dB)

      val dYPreviousIm = implicitly[CanTransformForConv[TransformType.COL_GRAD_2_IMAGE, (DenseMatrix[Double], ConvSize, Filter), DenseMatrix[Double]]]
        .transform((dYPrevious, by.preConvSize, by.param))

      (dYPreviousIm, grads)
    }

  implicit val convLayerCanRegularize = new CanRegularize[ConvLayer] {
    override def regu(foor: ConvLayer, regularizer: Option[Regularizer]): Double = {
      regularizer match {
        case None => 0.0
        case r: Regularizer => r.getReguCost(foor.param.w)
      }
    }
  }
}


/**
  * Created by mengpan on 2017/9/7.
  */
class DropoutLayer(override val numHiddenUnits: Int, val dropoutRate: Double)
  extends Layer with LayerLike[DropoutLayer] with DropoutActivator {

  /** Cache processed data */
  protected[estuary] var yPrevious: DenseMatrix[Double] = _

  def copyStructure: DropoutLayer = new DropoutLayer(numHiddenUnits, dropoutRate)
}

object DropoutLayer {
  def apply(numHiddenUnits: Int, dropoutRate: Double): DropoutLayer = {
    new DropoutLayer(numHiddenUnits, dropoutRate)
  }

  implicit val dropoutLayerCanSetParam: CanSetParam[DropoutLayer, None.type] = (_, _) => {}

  implicit val dropoutLayerCanExportParam: CanExportParam[DropoutLayer, None.type] = (_) => None

  implicit val dropoutLayerCanAutoInit: CanAutoInit[DropoutLayer] = (_, _) => {}

  implicit val dropoutLayerCanForward: CanForward[DropoutLayer, DenseMatrix[Double], DenseMatrix[Double]] =
    (input, by) => {
      by.yPrevious = input
      by.activate(input)
    }

  implicit val dropoutLayerCanForwardForPrediction: CanForward[DropoutLayer, ForPrediction[DenseMatrix[Double]], DenseMatrix[Double]] =
    (input, _) => input.input

  implicit val dropoutLayerCanBackward: CanBackward[DropoutLayer, DenseMatrix[Double], (DenseMatrix[Double], None.type)] =
    (input, by, regularizer) => {
      val filterMat = by.activateGrad(by.yPrevious)
      (input *:* filterMat, None)
    }

  implicit val dropoutLayerCanRegularize = new CanRegularize[DropoutLayer] {
    override def regu(foor: DropoutLayer, regularizer: Option[Regularizer]): Double = 0.0
  }
}


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




