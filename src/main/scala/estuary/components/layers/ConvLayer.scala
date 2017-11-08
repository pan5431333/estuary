package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Rand
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.ConvLayer.{ConvSize, Filter, FilterGrad, RichImageFeature}
import estuary.components.regularizer.Regularizer
import estuary.implicits._


trait ConvLayer extends Layer with Activator {
  protected val filter: Filter
  protected var preConvSize: ConvSize
  lazy protected val outputConvSize: ConvSize = calConvSize(preConvSize, filter)
  lazy val numHiddenUnits: Int = outputConvSize.dataLength
  lazy val previousHiddenUnits: Int = preConvSize.dataLength

  /** cache intermediate results to be used later */
  protected var yPrevious: DenseMatrix[Double] = _
  protected var z: DenseMatrix[Double] = _
  protected var y: DenseMatrix[Double] = _

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
    this.yPrevious = yPrevious
    val n = yPrevious.rows.toDouble
    val N = n.toInt
    z = DenseMatrix.zeros[Double](N, outputConvSize.dataLength)
    for (i <- (0 until N).par) {
      val ithRow = RichImageFeature(yPrevious(i, ::).t.copy.data, preConvSize)
      z(i, ::) := new DenseVector[Double](convolve(ithRow, filter).data).t
    }
    this.y = activationFuncEval(z)
    this.y
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
    val dZ = dYCurrent *:* activationGradEval(z)
    val (dYPrevious, filterGrads): (DenseMatrix[Double], FilterGrad) = convolveGrad(dZ, filter, yPrevious)
    (dYPrevious, filterGrads.toDenseMatrix)
  }

  def convolveGrad(dZ: DenseMatrix[Double], filter: ConvLayer.Filter, yPrevious: DenseMatrix[Double]): (DenseMatrix[Double], FilterGrad) = {
    val dYPrevious: Seq[RichImageFeature] = (0 until yPrevious.rows).par.map(_ => RichImageFeature.zeros(preConvSize)).seq
    val filterGrads: FilterGrad = new FilterGrad(filter)

    val n = dZ.rows.toDouble
    val N = n.toInt
    for (i <- (0 until N).par) {
      val gradData = dZ(i, ::).t.copy.data
      val grad = new RichImageFeature(gradData, outputConvSize)
      val yData = yPrevious(i, ::).t.copy.data
      val dYData = dYPrevious(i)
      val yP = new RichImageFeature(yData, preConvSize)
      val newYP = if (filter.pad == 0) yP else yP.pad(filter.pad, filter.pad, 0, 0.0)
      for {c <- 0 until outputConvSize.channel
           w <- 0 until outputConvSize.width
           h <- 0 until outputConvSize.height
      } {
        val heightRange = (h * filter.stride) until (h * filter.stride + filter.size)
        val widthRange = (w * filter.stride) until (w * filter.stride + filter.size)
        val oldChannelRange = 0 until preConvSize.channel

        dYData.+=(heightRange, widthRange, oldChannelRange, (filter.w(c) * grad.get(h, w, c)).data)
        filterGrads.addDW(c, (newYP.slice(heightRange, widthRange, oldChannelRange) * grad.get(h, w, c)).data)
        filterGrads.addDB(c, grad.get(h, w, c))
      }
    }

    (dYPrevious.toDenseMatrix, filterGrads)
  }

  def convolve(feature: ConvLayer.RichImageFeature, filter: ConvLayer.Filter): RichImageFeature = {
    val outputConvSize = calConvSize(feature.convSize, filter)
    val newFeature = if (filter.pad == 0) feature else feature.pad(filter.pad, filter.pad, 0, 0.0)
    val data = (for {c <- (0 until outputConvSize.channel).par
                     w <- (0 until outputConvSize.width).par
                     h <- (0 until outputConvSize.height).par
    } yield {
      val heightRange = (h * filter.stride) until (h * filter.stride + filter.size)
      val widthRange = (w * filter.stride) until (w * filter.stride + filter.size)
      val channelRange = 0 until newFeature.convSize.channel
      (newFeature.slice(heightRange, widthRange, channelRange) *:* filter.w(c)).data.par.sum + filter.b(c)
    }).toArray
    RichImageFeature(data, outputConvSize)
  }

  def calConvSize(pre: ConvLayer.ConvSize, filter: ConvLayer.Filter): ConvSize = {
    val outHeight = calOutDimension(preConvSize.height, filter.size, filter.pad, filter.stride)
    val outWidth = calOutDimension(preConvSize.width, filter.size, filter.pad, filter.stride)
    val outChannel = filter.newChannel
    ConvSize(outHeight, outWidth, outChannel)
  }

  protected def calOutDimension(inputDim: Int, filterSize: Int, pad: Int, stride: Int): Int = {
    (inputDim + 2 * pad - filterSize) / stride + 1
  }
}

object ConvLayer {

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
      data(channel * (convSize.width * convSize.height) + width * convSize.height + height)
    }

    def update(height: Int, width: Int, channel: Int, newVal: Double): Unit = {
      data(channel * convSize.channel + width * convSize.height + height) = newVal
    }

    def update(newData: Array[Double]): RichImageFeature = {
      require(size == newData.length, s"Unmatched index range and data's length: ($size, ${newData.length})")
      RichImageFeature(newData, convSize)
    }

    def +=(height: Int, width: Int, channel: Int, newVal: Double): Unit = {
      data(channel * convSize.channel + width * convSize.height + height) += newVal
    }

    def +=(newData: Array[Double]): Unit = {
      require(size == newData.length, s"Unmatched index range and data's length: ($size, ${newData.length})")
      +=(0 until convSize.height, 0 until convSize.width, 0 until convSize.channel, newData)
    }

    def update(heightRange: Range, widthRange: Range, channelRange: Range, newData: Array[Double]): Unit = {
      require(heightRange.size * widthRange.size * channelRange.size == newData.length, s"Unmatched index range and data's length: (${heightRange.size * widthRange.size * channelRange.size}, ${newData.length})")

      def getNewDataIndex(h: Int, w: Int, c: Int): Double = {
        val index = c * channelRange.size + w * widthRange.size + h
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
        val index = c * channelRange.size + w * widthRange.size + h
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
      new RichImageFeature(data.par.map(_ * d).seq.toArray, convSize)
    }

    def pad(h: Int, w: Int, c: Int, value: Double): RichImageFeature = {
      val newConvSize = ConvSize(convSize.height + 2*h, convSize.width + 2*w, convSize.channel+2*c)
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
