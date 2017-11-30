package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.ConvLayer.{ConvSize, Filter, calConvSize}
import estuary.components.regularizer.Regularizer


trait ConvLayer extends Layer with Activator {
  /** Layer parameters */
  protected val filter: Filter
  protected var preConvSize: ConvSize

  /**Inferred layer structure*/
  lazy protected val outputConvSize: ConvSize = calConvSize(preConvSize, filter)
  lazy val numHiddenUnits: Int = outputConvSize.dataLength
  lazy val previousHiddenUnits: Int = preConvSize.dataLength

  /** cache intermediate results to be used later */
  protected var yPrevious: DenseMatrix[Double] = _
  protected var yPreviousIm2Col: DenseMatrix[Double] = _
  protected var z: DenseMatrix[Double] = _
  protected var zIm2Col: DenseMatrix[Double] = _
  protected var y: DenseMatrix[Double] = _
  protected var filterMatrix: DenseMatrix[Double] = _
  protected var filterBias: DenseVector[Double] = _

  def setPreConvSize(pre: ConvSize): this.type = {
    this.preConvSize = pre
    this
  }

  def setPreConvSize(preHeight: Int, preWidth: Int, preChannel: Int): this.type = {
    setPreConvSize(ConvSize(preHeight, preWidth, preChannel))
  }

  /**
    * Forward propagation of current layer.
    *
    * @param yPrevious Output of previous layer, of the shape (n, d(l-1)), where
    *                  n: #training examples,
    *                  d(l-1): #hidden units in previous layer L-1.
    * @return Output of this layer, of the shape (n, d(l)), where
    *         n: #training examples,
    *         d(l): #hidden units in current layer L.
    */
  override def forward(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    checkInputValidity(yPrevious, preConvSize)

    this.yPreviousIm2Col = ConvLayer.im2col(yPrevious, preConvSize, filter)

    val filterWeightsAndBias = filter.toIm2Col
    this.filterMatrix = filterWeightsAndBias._1 // shape (size * size * oldChannel, newChannel)
    this.filterBias = filterWeightsAndBias._2

    zIm2Col = this.yPreviousIm2Col * this.filterMatrix + DenseVector.ones[Double](this.yPreviousIm2Col.rows) * this.filterBias.t
    z = ConvLayer.col2im(zIm2Col, outputConvSize)
    y = activate(z)
    y
  }

  def checkInputValidity(yPrevious: DenseMatrix[Double], pre: ConvLayer.ConvSize): Unit = {
    val cols = yPrevious.cols
    val convSize = pre.height * pre.width * pre.channel
    assert(cols == convSize, s"Input data's cols not equal to convSize, ($cols != ${pre.height} * ${pre.width} * ${pre.channel})")
  }

  override def init(initializer: WeightsInitializer): DenseMatrix[Double] = {
    filter.init(initializer)
    filter.toDenseMatrix
  }

  override def getReguCost(regularizer: Option[Regularizer]): Double = {
    regularizer match {
      case None => 0.0
      case Some(regu) => regu.getReguCost(filter.w.toDenseMatrix)
    }
  }

  override def forwardForPrediction(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    forward(yPrevious)
  }

  override def setParam(param: DenseMatrix[Double]): Unit = {
    filter.fromDenseMatrix(param)
  }

  /**
    * Backward propagation of current layer.
    *
    * @param dYCurrent Gradients of current layer's output, DenseMatrix of shape (n, d(l))
    *                  where n: #training examples,
    *                  d(l): #hidden units in current layer L.
    * @return (dYPrevious, grads), where dYPrevious is gradients for output of previous
    *         layer; grads is gradients of current layer's parameters, i.e. for layer
    *         without batchNorm, parameters are w and b, for layers with batchNorm,
    *         parameters are w, alpha and beta.
    */
  override def backward(dYCurrent: DenseMatrix[Double], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val dZ = dYCurrent *:* activateGrad(z)
    val dZCol = ConvLayer.imGrad2Col(dZ, outputConvSize)
    val dYPrevious = dZCol * this.filterMatrix.t
    val n = dYCurrent.rows.toDouble
    val dW = this.yPreviousIm2Col.t * dZCol / n
    val dB = (dZCol.t * DenseVector.ones[Double](dZCol.rows)).toDenseMatrix / n
    val grads = DenseMatrix.vertcat(dW, dB)
    val dYPreviousIm = ConvLayer.colGrad2Im(dYPrevious, preConvSize, filter)

    (dYPreviousIm, grads)
  }
}

object ConvLayer {

  protected def calOutDimension(inputDim: Int, filterSize: Int, pad: Int, stride: Int): Int = {
    (inputDim + 2 * pad - filterSize) / stride + 1
  }

  def calConvSize(pre: ConvLayer.ConvSize, filter: ConvLayer.Filter): ConvSize = {
    val outHeight = calOutDimension(pre.height, filter.size, filter.pad, filter.stride)
    val outWidth = calOutDimension(pre.width, filter.size, filter.pad, filter.stride)
    val outChannel = filter.newChannel
    ConvSize(outHeight, outWidth, outChannel)
  }

  def colIndex(h: Int, w: Int, c: Int)(implicit height: Int, width: Int): Int = c * height * width + w * height + h

  def im2col(y: DenseMatrix[Double], pre: ConvSize, filter: Filter): DenseMatrix[Double] = {
    val outSize = calConvSize(pre, filter)
    val n = y.rows
    val N = n.toInt
    val reshapedRow = N * outSize.height * outSize.width
    val reshapedCol = filter.size * filter.size * filter.oldChannel
    val res = DenseMatrix.zeros[Double](reshapedRow, reshapedCol)
    for {r <- (0 until reshapedRow).par
         c <- (0 until reshapedCol).par
    } {
      val en = r / (outSize.height * outSize.width)
      val er = r % (outSize.height * outSize.width)
      val ecol = c % (filter.size * filter.size)
      val echannel = c / (filter.size * filter.size)
      val or = (er / outSize.width) * filter.stride + ecol / filter.size
      val ocol = (er % outSize.width) * filter.stride + ecol % filter.size
      val colindex = colIndex(or, ocol, echannel)(pre.height, pre.width)
      val eres = y(en, colindex)
      res(r, c) = eres
    }

    res
  }

  def col2im(y: DenseMatrix[Double], oConvSize: ConvSize): DenseMatrix[Double] = {
    require(y.cols == oConvSize.channel)

    val bulkLen = oConvSize.height * oConvSize.width
    val n = y.rows.toInt / bulkLen.toInt
    val res = DenseMatrix.zeros[Double](n, bulkLen * oConvSize.channel)
    for (i <- (0 until n).par) {
      val startIndex = i * bulkLen
      val endIndex = (i + 1) * bulkLen
      val bulk = y(startIndex until endIndex, ::)
      res(i, ::) := new DenseVector[Double](bulk.copy.data).t
    }
    res
  }

  def imGrad2Col(dZ: DenseMatrix[Double], convSize: ConvSize): DenseMatrix[Double] = {
    require(dZ.cols == convSize.height * convSize.width * convSize.channel)

    val N = dZ.rows.toInt
    val res = DenseMatrix.zeros[Double](convSize.height * convSize.width * dZ.rows, convSize.channel)

    for (i <- (0 until N).par) {
      val startIndex = i * (convSize.height * convSize.width)
      val endIndex = (i + 1) * (convSize.height * convSize.width)
      val rowRange = startIndex until endIndex
      res(rowRange, ::) := DenseMatrix.create[Double](rowRange.size, convSize.channel, dZ(i, ::).t.copy.data)
    }

    res
  }

  def colGrad2Im(dYPrevious: DenseMatrix[Double], pre: ConvSize, filter: Filter): DenseMatrix[Double] = {
    val outSize = calConvSize(pre, filter)
    val N = dYPrevious.rows.toInt / (outSize.height * outSize.width)

    val res = DenseMatrix.zeros[Double](N, pre.height * pre.width * pre.channel)

    for {r <- (0 until dYPrevious.rows).par
         c <- (0 until dYPrevious.cols).par
    } {
      val en = r / (outSize.height * outSize.width)
      val er = r % (outSize.height * outSize.width)
      val ecol = c % (filter.size * filter.size)
      val echannel = c / (filter.size * filter.size)
      val or = (er / outSize.width) * filter.stride + ecol / filter.size
      val ocol = (er % outSize.width) * filter.stride + ecol % filter.size
      val colindex = colIndex(or, ocol, echannel)(pre.height, pre.width)
      res(en, colindex) += dYPrevious(r, c)
    }

    res
  }

  case class Filter(size: Int, pad: Int, stride: Int, oldChannel: Int, newChannel: Int) {
    val w: DenseMatrix[Double] = DenseMatrix.zeros[Double](size * size * oldChannel, newChannel)
    val b: DenseVector[Double] = DenseVector.zeros[Double](newChannel)
    val matrixShape: (Int, Int) = (w.rows + 1, w.cols)

    def init(initializer: WeightsInitializer): this.type = {
      val params = initializer.init(matrixShape._1, matrixShape._2)
      fromDenseMatrix(params)
      this
    }

    def toIm2Col: (DenseMatrix[Double], DenseVector[Double]) = {
      (w, b)
    }

    def toDenseMatrix: DenseMatrix[Double] = {
      DenseMatrix.vertcat(w, b.toDenseMatrix)
    }

    def fromDenseMatrix(m: DenseMatrix[Double]): Unit = {
      assert(m.rows == matrixShape._1)
      assert(m.cols == matrixShape._2)

      for (i <- 0 until w.rows) {
        w(i, ::) := m(i, ::)
      }

      b := m(w.rows, ::).t
    }

    override def toString: String = {
      s"""(size: $size, pad: $pad, stride: $stride, oldChannel: $oldChannel, newChannel: $newChannel)"""
    }
  }

  case class ConvSize(height: Int, width: Int, channel: Int) {
    def ==(that: ConvSize): Boolean = {
      (height == that.height) && (width == that.width) && (channel == that.channel)
    }

    def contains(height: Int, width: Int, channel: Int): Boolean = {
      height <= this.height && width <= this.width && channel <= this.channel
    }

    val dataLength: Int = height * width * channel

    override def toString: String = s"""(height: $height, width: $width, channel: $channel)"""
  }
}
