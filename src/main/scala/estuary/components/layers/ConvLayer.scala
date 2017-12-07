package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.ConvLayer.{ConvSize, Filter, calConvSize}
import estuary.components.regularizer.Regularizer
import estuary.components.support._


trait ConvLayer extends Layer with LayerLike[ConvLayer] with Activator {

  /** LayerLike parameters */
  protected[estuary] var preConvSize: ConvSize = _
  def setPreConvSize(pre: ConvSize): this.type = {
    this.preConvSize = pre
    this
  }
  def setPreConvSize(preHeight: Int, preWidth: Int, preChannel: Int): this.type = {
    setPreConvSize(ConvSize(preHeight, preWidth, preChannel))
  }

  val param: Filter

  override def hasParams = true

  /** Inferred layer structure */
  protected[estuary] lazy val outputConvSize: ConvSize = calConvSize(preConvSize, param)
  lazy val numHiddenUnits: Int = outputConvSize.dataLength
  lazy val previousHiddenUnits: Int = preConvSize.dataLength

  /** cache intermediate results to be used later */
  protected[estuary] var yPrevious: DenseMatrix[Double] = _
  protected[estuary] var yPreviousIm2Col: DenseMatrix[Double] = _
  protected[estuary] var z: DenseMatrix[Double] = _
  protected[estuary] var zIm2Col: DenseMatrix[Double] = _
  protected[estuary] var y: DenseMatrix[Double] = _
  protected[estuary] var filterMatrix: DenseMatrix[Double] = _
  protected[estuary] var filterBias: DenseVector[Double] = _
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
