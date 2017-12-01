package estuary.components.support

import breeze.linalg.{DenseMatrix, DenseVector}
import estuary.components.layers.ConvLayer.{ConvSize, Filter, calConvSize}

trait CanTransformForConv[TranType, From, To] {
  def transform(from: From): To
}

object CanTransformForConv {
  implicit val filterCanTransformForConv: CanTransformForConv[TransformType.FILTER_TO_COL, Filter, (DenseMatrix[Double], DenseVector[Double])] =
    (from) => (from.w, from.b)

  implicit val im2colCanTransformForConv: CanTransformForConv[TransformType.IMAGE_TO_COL, (DenseMatrix[Double], ConvSize, Filter), DenseMatrix[Double]] = {
    case (y: DenseMatrix[Double], pre: ConvSize, filter: Filter) =>
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
        val colindex = pre.linearIndex(or, ocol, echannel)
        val eres = y(en, colindex)
        res(r, c) = eres
      }

      res
  }

  implicit val col2imCanTransformForConv: CanTransformForConv[TransformType.COL_TO_IMAGE, (DenseMatrix[Double], ConvSize), DenseMatrix[Double]] = {
    case (y: DenseMatrix[Double], oConvSize: ConvSize) =>
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

  implicit val imGrad2ColCanTransformForConv: CanTransformForConv[TransformType.IMAGE_GRAD_2_COL, (DenseMatrix[Double], ConvSize), DenseMatrix[Double]] = {
    case (dZ: DenseMatrix[Double], convSize: ConvSize) =>
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

  implicit val colGrad2ImCanTransformForConv: CanTransformForConv[TransformType.COL_GRAD_2_IMAGE, (DenseMatrix[Double], ConvSize, Filter), DenseMatrix[Double]] = {
    case (dYPrevious: DenseMatrix[Double], pre: ConvSize, filter: Filter) =>
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
        val colindex = pre.linearIndex(or, ocol, echannel)
        res(en, colindex) += dYPrevious(r, c)
      }

      res
  }
}
