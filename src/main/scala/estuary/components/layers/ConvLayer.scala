package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Rand
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.ConvLayer.{ConvSize, Filter, FilterGrad, RichImageFeature, calConvSize}
import estuary.components.regularizer.Regularizer
import estuary.implicits._


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
  protected var numExamples: Int = 0

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

    this.numExamples = yPrevious.rows
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
    val dW = this.yPreviousIm2Col.t * dZCol
    val dB = (dZCol.t * DenseVector.ones[Double](dZCol.rows)).toDenseMatrix
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
    for {r <- 0 until reshapedRow
         c <- 0 until reshapedCol
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

    for {r <- 0 until dYPrevious.rows
         c <- 0 until dYPrevious.cols
    } {
      val en = r / (outSize.height * outSize.width)
      val er = r % (outSize.height * outSize.width)
      val ecol = c % (filter.size * filter.size)
      val echannel = c / (filter.size * filter.size)
      val or = (er / outSize.width) * filter.stride + ecol / filter.size
      val ocol = (er % outSize.width) * filter.stride + ecol % filter.size
      val colindex = colIndex(or, ocol, echannel)(pre.height, pre.width)
      res(en, colindex) = dYPrevious(r, c)
    }

    res
  }

  case class Filter(size: Int, pad: Int, stride: Int, oldChannel: Int, newChannel: Int) {
    var w: Seq[RichImageFeature] = (0 until newChannel).par.map(_ => RichImageFeature.zeros(size, size, oldChannel)).seq
    var b: Array[Double] = (0 until newChannel).map(_ => 0.0).toArray
    val matrixShape: (Int, Int) = (size * size * oldChannel + 1, newChannel)

    def init(initializer: WeightsInitializer): this.type = {
      val params = initializer.init(matrixShape._1, matrixShape._2)
      fromDenseMatrix(params)
      this
    }

    def update(newFilter: Filter): this.type = {
      require(this == newFilter, s"update by a filter of different size! ${this} updated by $newFilter")
      w = newFilter.w
      b = newFilter.b
      this
    }

    def toIm2Col: (DenseMatrix[Double], DenseVector[Double]) = {
      val resW = DenseMatrix.zeros[Double](size * size * oldChannel, newChannel)
      for (j <- 0 until resW.cols) {
        resW(::, j) := w(j).toDenseVector
      }

      val resB = new DenseVector[Double](b)
      (resW, resB)
    }

    /**
      * Almost identical to toIm2Col, except that w and b are concatenated vertically.
      * @return
      */
    def toDenseMatrix: DenseMatrix[Double] = {
      val res = DenseMatrix.zeros[Double](matrixShape._1, matrixShape._2)
      for (((w_, b_), j) <- w.zip(b).zipWithIndex.par) {
        res(0 until w_.size, j) := w_.toDenseVector
        res(w_.size, j) = b_
      }
      res
    }

    def fromDenseMatrix(m: DenseMatrix[Double]): Unit = {
      val (ws, bs) = (0 until m.cols).par.map { j =>
        val data = m(::, j).copy
        (w(j).update(data(0 until data.length - 1).copy.data), data.data.last)
      }.seq.unzip
      w = ws
      b = bs.toArray
    }

    def ==(that: Filter): Boolean = {
      size == that.size && pad == that.pad && stride == that.stride && oldChannel == that.oldChannel && newChannel == that.newChannel
    }

    override def toString: String = {
      s"""(size: $size, pad: $pad, stride: $stride, oldChannel: $oldChannel, newChannel: $newChannel)"""
    }
  }

  class FilterGrad(filter: Filter) {
    val size: Int = filter.size
    val pad: Int = filter.pad
    val stride: Int = filter.stride
    val oldChannel: Int = filter.oldChannel
    val newChannel: Int = filter.newChannel
    val dW: Seq[RichImageFeature] = (0 until newChannel).par.map(_ => RichImageFeature.zeros(size, size, oldChannel)).seq
    val dB: Array[Double] = (0 until newChannel).map(_ => 0.0).toArray

    def updateDW(height: Int, width: Int, oldChannel: Int, newChannel: Int, newW: Double): Unit = {
      dW(newChannel).update(height, width, oldChannel, newW)
    }

    def addDW(height: Int, width: Int, oldChannel: Int, newChannel: Int, newW: Double): Unit = {
      dW(newChannel).+=(height, width, oldChannel, newW)
    }

    def updateDW(newChannel: Int, newData: Array[Double]): Unit = {
      require(newData.length == dW.head.size, s"newData.length not equal to the filter's range: (${newData.length}, ${dW.head.size})")
      dW(newChannel) update newData
    }

    def addDW(newChannel: Int, newData: Array[Double]): Unit = {
      require(newData.length == dW.head.size, s"newData.length not equal to the filter's range: (${newData.length}, ${dW.head.size})")
      dW(newChannel) += newData
    }

    def updateDB(newChannel: Int, newB: Double): Unit = {
      dB(newChannel) = newB
    }

    def addDB(newChannel: Int, newB: Double): Unit = {
      dB(newChannel) += newB
    }

    def toDenseMatrix: DenseMatrix[Double] = {
      val res = DenseMatrix.zeros[Double](dW.head.size + 1, dW.size)
      for (((w_, b_), j) <- dW.zip(dB).zipWithIndex.par) {
        res(0 until w_.size, j) := w_.toDenseVector
        res(w_.size, j) = b_
      }
      res
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

  case class RichImageFeature(data: Array[Double], convSize: ConvSize) {
    require(data.length == convSize.height * convSize.width * convSize.channel, s"unmatched data and convSize (${data.length}, $convSize)")
    val size: Int = convSize.height * convSize.width * convSize.channel

    private def getDataIndex = (h: Int, w: Int, c: Int) => {
      c * (convSize.height * convSize.width) + w * convSize.height + h
    }


    def slice(heightRange: Range, widthRange: Range, channelRange: Range): RichImageFeature = {
      val data = (for {h <- heightRange.par
                       w <- widthRange.par
                       c <- channelRange.par
      } yield get(h, w, c)).toArray
      val newConvSize = ConvSize(heightRange.size, widthRange.size, channelRange.size)
      RichImageFeature(data, newConvSize)
    }

    def get(height: Int, width: Int, channel: Int): Double = {
      require(convSize.contains(height, width, channel), s"(height = $height, width = $width, channel = $channel) out of index bound")
      data(getDataIndex(height, width, channel))
    }

    def update(height: Int, width: Int, channel: Int, newVal: Double): Unit = {
      data(getDataIndex(height, width, channel)) = newVal
    }

    def update(newData: Array[Double]): RichImageFeature = {
      require(size == newData.length, s"Unmatched index range and data's length: ($size, ${newData.length})")
      RichImageFeature(newData, convSize)
    }

    def +=(height: Int, width: Int, channel: Int, newVal: Double): Unit = {
      data(getDataIndex(height, width, channel)) += newVal
    }

    def +=(newData: Array[Double]): Unit = {
      require(size == newData.length, s"Unmatched index range and data's length: ($size, ${newData.length})")
      +=(0 until convSize.height, 0 until convSize.width, 0 until convSize.channel, newData)
    }

    def update(heightRange: Range, widthRange: Range, channelRange: Range, newData: Array[Double]): Unit = {
      require(heightRange.size * widthRange.size * channelRange.size == newData.length, s"Unmatched index range and data's length: (${heightRange.size * widthRange.size * channelRange.size}, ${newData.length})")

      def getNewDataIndex(h: Int, w: Int, c: Int): Double = {
        val index = c * (heightRange.size * widthRange.size) + w * heightRange.size + h
        newData(index)
      }

      for {(h, hi) <- heightRange.zipWithIndex.par
           (w, wi) <- widthRange.zipWithIndex.par
           (c, ci) <- channelRange.zipWithIndex.par
      } {
        update(h, w, c, getNewDataIndex(hi, wi, ci))
      }
    }

    def +=(heightRange: Range, widthRange: Range, channelRange: Range, newData: Array[Double]): Unit = {
      require(heightRange.size * widthRange.size * channelRange.size == newData.length, s"Unmatched index range and data's length: (${heightRange.size * widthRange.size * channelRange.size}, ${newData.length})")

      def getNewDataIndex(h: Int, w: Int, c: Int): Double = {
        val index = c * (heightRange.size * widthRange.size) + w * heightRange.size + h
        newData(index)
      }

      for {(h, hi) <- heightRange.zipWithIndex.par
           (w, wi) <- widthRange.zipWithIndex.par
           (c, ci) <- channelRange.zipWithIndex.par
      } {
        +=(h, w, c, getNewDataIndex(hi, wi, ci))
      }
    }

    def *:*(that: RichImageFeature): RichImageFeature = {
      require(convSize == that.convSize, s"RichImageFeatures $convSize and ${that.convSize} with different shape multiplied")
      val data = this.data.zip(that.data).par.map { case (a, b) => a * b }.seq.toArray
      RichImageFeature(data, convSize)
    }

    def *(d: Double): RichImageFeature = {
      new RichImageFeature(data.map(_ * d), convSize)
    }

    def pad(h: Int, w: Int, c: Int, value: Double): RichImageFeature = {
      val newConvSize = ConvSize(convSize.height + 2 * h, convSize.width + 2 * w, convSize.channel + 2 * c)
      val padded = RichImageFeature(DenseVector.ones[Double](newConvSize.dataLength).data, newConvSize) * value
      padded.update(h until convSize.height + h, w until convSize.width + w, c until convSize.channel + c, data)
      padded
    }

    def toDenseVector: DenseVector[Double] = {
      new DenseVector[Double](data)
    }

  }

  object RichImageFeature {
    def rand(height: Int, width: Int, channel: Int, rand: Rand[Double]): RichImageFeature = {
      RichImageFeature(DenseVector.rand[Double](height * width * channel, rand).data, ConvSize(height, width, channel))
    }

    def zeros(height: Int, width: Int, channel: Int): RichImageFeature = {
      RichImageFeature(DenseVector.zeros[Double](height * width * channel).data, ConvSize(height, width, channel))
    }

    def zeros(convSize: ConvSize): RichImageFeature = {
      zeros(convSize.height, convSize.width, convSize.channel)
    }

    def rand(convSize: ConvSize, rand: Rand[Double]): RichImageFeature = {
      this.rand(convSize.height, convSize.width, convSize.channel, rand)
    }
  }

}
